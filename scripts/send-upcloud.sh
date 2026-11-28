#!/usr/bin/env bash
set -euo pipefail
umask 077

environment="${1:-}"
revision="${2:-}"
image="${3:-}"
[[ "$environment" == staging || "$environment" == production ]]
[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
[[ "$image" =~ ^ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$ ]]
[[ "$DEPLOY_HOST" =~ ^[a-zA-Z0-9.-]+$ ]]
[[ "$DEPLOY_USER" =~ ^[a-z_][a-z0-9_-]*$ ]]
test -n "$DEPLOY_SSH_KEY"
test -n "$DEPLOY_SSH_KNOWN_HOSTS"

ssh_dir="$(mktemp -d)"
trap 'rm -rf "$ssh_dir"' EXIT
printf '%s\n' "$DEPLOY_SSH_KEY" > "$ssh_dir/key"
printf '%s\n' "$DEPLOY_SSH_KNOWN_HOSTS" > "$ssh_dir/known_hosts"
tar -czf - deploy/compose.yml deploy/compose.upcloud.yml scripts/deploy-upcloud.sh | \
    ssh -T -i "$ssh_dir/key" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile="$ssh_dir/known_hosts" "$DEPLOY_USER@$DEPLOY_HOST" \
    "set -eu; release=\$(mktemp -d); trap 'rm -rf \"\$release\"' EXIT; tar -xzf - -C \"\$release\"; bash \"\$release/scripts/deploy-upcloud.sh\" $environment $revision $image \"\$release/deploy/compose.yml\""
