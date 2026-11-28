#!/usr/bin/env bash
set -euo pipefail
umask 077

environment="${1:-}"
revision="${2:-}"
image="${3:-}"
manifest="${4:-}"
edge_manifest="$(dirname -- "$manifest")/compose.upcloud.yml"
[[ "$environment" == staging || "$environment" == production ]]
[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
[[ "$image" =~ ^ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$ ]]
[[ -f "$manifest" ]]
[[ -f "$edge_manifest" ]]

root="${AGENTMILIEU_ROOT:-/opt/agentmilieu}/$environment"
cd "$root"
[[ -f .env ]]
mkdir -p backups
exec 9> .deploy.lock
flock -n 9
old_image=""
if [[ -s current-image && -s current-compose.yml ]]; then
    old_image="$(cat current-image)"
fi
export AGENTMILIEU_IMAGE="$image"
compose=(docker compose --project-name "agentmilieu-$environment" --env-file "$root/.env")
release=("${compose[@]}" -f "$manifest" -f "$edge_manifest")
"${release[@]}" config --quiet
docker network inspect agentkeireki-edge > /dev/null
docker pull "$image"
test "$(docker image inspect -f '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "$image")" = "$revision"
"${release[@]}" up -d --wait --wait-timeout 90 db
backup="backups/postgres-$(date -u +%Y%m%dT%H%M%S)-${revision:0:12}.dump"
if [[ -n "$old_image" ]]; then
    test ! -e "$backup"
    "${release[@]}" exec -T db pg_dump -Fc -U agentmilieu agentmilieu > "$backup.partial"
    test -s "$backup.partial"
    "${release[@]}" exec -T db pg_restore -l < "$backup.partial" > /dev/null
    mv -- "$backup.partial" "$backup"
fi

rollback() {
    if [[ -n "$old_image" ]]; then
        echo "Deployment failed. Restoring the previous application image." >&2
        previous=("${compose[@]}" -f "$root/current-compose.yml")
        if [[ -s current-compose.upcloud.yml ]]; then
            previous+=(-f "$root/current-compose.upcloud.yml")
        fi
        AGENTMILIEU_IMAGE="$old_image" "${previous[@]}" up -d --wait --wait-timeout 120 app || true
    fi
}
trap rollback ERR
"${release[@]}" up -d --wait --wait-timeout 180 app
binding="$("${release[@]}" port app 8080)"
curl --fail --silent --show-error --retry 5 --retry-delay 2 "http://$binding/healthz" > /dev/null
curl --fail --silent --show-error "http://$binding/" > /dev/null
curl --fail --silent --show-error "http://$binding/app" > /dev/null
curl --fail --silent --show-error --head "http://$binding/downloads/AgentMilieu-Setup.exe" > /dev/null
install -m 600 "$manifest" current-compose.yml.next
install -m 600 "$edge_manifest" current-compose.upcloud.yml.next
printf '%s\n' "$image" > current-image.next
mv -- current-compose.yml.next current-compose.yml
mv -- current-compose.upcloud.yml.next current-compose.upcloud.yml
mv -- current-image.next current-image
printf 'Verified %s on %s (%s)\n' "$revision" "$environment" "$image"
