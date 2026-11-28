# AgentMilieu deployment

AgentMilieu serves the landing page, web workspace, shared account API, and the
Windows x64 installer from one release image. PostgreSQL stores Google accounts,
sessions, wallet transactions, and reviewed packs. Each environment has its own
database and credentials.

## Configure the UpCloud environments

1. Reuse the Docker Engine, Compose v2 plugin, and `naninithin` SSH account on
   AgentKeireki's UpCloud server. The deploy account needs access to Docker
   without interactive sudo. Create the AgentMilieu directories using that
   account as their owner. For a private GHCR image, run `docker login ghcr.io`
   as the deploy user with a read-only package credential.

   ```sh
   sudo install -d -m 0750 -o naninithin -g "$(id -gn naninithin)" \
     /opt/agentmilieu/staging /opt/agentmilieu/production
   ```

2. Copy [the environment template](product.env.example) to `.env` in each
   directory and restrict access to the deploy user with mode `0600`. Generate a
   different PostgreSQL password and at least 32 random characters for
   `SESSION_SECRET` for each environment. Populate `DATABASE_URL` with the same
   PostgreSQL password. Leave both PostgreSQL instances inside their Compose networks.
3. Use loopback port `8080` for production and `8086` for staging. The deployment
   script uses these ports to check the containers from the host. Set the exact
   HTTPS origin in each `.env` without a trailing slash. This guide uses
   `https://staging.agentmilieu.naninithin.me` for staging and
   `https://agentmilieu.naninithin.me` for production.
4. Create Google OAuth credentials of type **Web application** and register
   `<PUBLIC_URL>/auth/google/callback` for each environment. Production uses
   `https://agentmilieu.naninithin.me/auth/google/callback`. The Windows app uses
   the same browser sign-in flow and displays a verification code.
   [Google OpenID Connect](https://developers.google.com/identity/openid-connect/openid-connect)
5. Enter your Mollie test API key in staging and your live profile key in
   production. The application sends a payment-specific webhook URL under
   `<PUBLIC_URL>/api/mollie/webhook/` when it creates each checkout. Mollie must
   be able to reach it without browser authentication.
   [Mollie Payments API](https://docs.mollie.com/reference/create-payment)
6. To enable **AgentMilieu managed** compression, set `MANAGED_PROVIDER_KIND`
   to `openai` or `anthropic`, set the model and key, and optionally set a custom
   HTTPS API base URL. Users provide their own compatible keys in either client,
   including Mistral through its OpenAI-compatible chat completions API. Custom
   endpoints must be reachable as public HTTPS services from UpCloud.
   Extractive generation is available without a model API key.

The [UpCloud Compose override](compose.upcloud.yml) joins the two AgentMilieu app
containers to the existing `agentkeireki-edge` network. Its aliases come from the
Compose project name. Add these routes to AgentKeireki's public Caddyfile using
its manual infrastructure procedure. Add the staging route and DNS record
before the first push to `main` and the production route during cutover.
[AgentKeireki runbook](https://github.com/NANInithin/AgentKeireki/blob/main/docs/upcloud-staging-production-runbook.md)

```caddyfile
agentmilieu.naninithin.me {
    reverse_proxy agentmilieu-production-app:8080
}
staging.agentmilieu.naninithin.me {
    reverse_proxy agentmilieu-staging-app:8080
}
```

## Set up GitHub Actions

Create GitHub environments called **staging** and **production**. Use these
variables and secrets in both environments.

| GitHub setting | Name | Value |
| --- | --- | --- |
| Variable | `UPCLOUD_TAILSCALE_HOST` | UpCloud Tailscale hostname or IP |
| Variable | `UPCLOUD_SSH_USER` | `naninithin` |
| Variable | `PUBLIC_URL` | Public staging HTTPS origin, required by the staging workflow |
| Secret | `TS_OAUTH_CLIENT_ID` | Tailscale federated identity client ID |
| Secret | `TS_AUDIENCE` | Audience for GitHub workload identity federation |
| Secret | `DEPLOY_SSH_KEY` | Private SSH deployment key |
| Secret | `DEPLOY_SSH_KNOWN_HOSTS` | Verified SSH host key entry for that exact host |

Use a dedicated AgentMilieu SSH key. Add its public key to
`/home/naninithin/.ssh/authorized_keys`. Give the Tailscale identity the
`tag:github-deploy` tag and permit it to reach the server's SSH port. For
`NANInithin/AgentMilieu`, the default immutable OIDC subject pattern is
`repo:NANInithin@77537690/AgentMilieu@1457815537:environment:*`. Use the custom
claim `ref=refs/heads/main` and allow deployments from `main` in both GitHub
environments. Configure the production environment's reviewer policy if you
require a second approval.

A push to `main` builds and tests the app, packages the Windows installer, and
publishes the release to GHCR. Successful CI automatically deploys that image
digest to staging. The staging job checks PostgreSQL, the public HTTPS health
endpoint, Google and Mollie configuration, and the public installer response.
Use the staging domain to verify Google sign-in, a test Mollie payment, and
approval/export before promotion.

Run **Promote production** from `main` with the staging commit SHA and the
confirmation string `deploy-production`. The SHA must still be the tip of
`main` and have a successful staging release artifact. The workflow deploys the
same image digest. Staging release metadata is retained for 30 days. PostgreSQL
snapshots for existing releases are saved in each environment's `backups/`
directory. Keep off-host backups and manage snapshot retention on the server.

For the initial cutover, add the production Caddy route, switch the production
DNS record to UpCloud, and dispatch the production workflow. After the public
HTTPS checks pass, remove the custom domain from GitHub Pages settings.

## Local builds

Build a web-only image with `docker build --target development -t agentmilieu:dev .`.
Copy the environment template to a local ignored `.env.product` file, set
`PUBLIC_URL=http://localhost:8080`, and provide the local database password and
session secret. Start the stack with Compose.

```sh
AGENTMILIEU_IMAGE=agentmilieu:dev docker compose \
  --project-name agentmilieu-local --env-file .env.product -f deploy/compose.yml up -d --wait
```

The local workspace is at `http://localhost:8080/app`. Google sign-in requires
the exact localhost callback in Google Cloud. Local Mollie checkout can refresh
on the return page. For webhook testing use the public staging host with a
Mollie test key.

On Windows with Python 3.12 x64 and NSIS 3, install
`desktop/requirements-build.txt` and run `desktop/build.ps1` from PowerShell.
The installer is written to `dist/windows/AgentMilieu-Setup.exe`. The Docker
release target includes that file in the public download. CI produces an
unsigned installer. To test the same installed desktop build against staging,
launch `AgentMilieu.exe --server https://<your-staging-host>`.

## Review and data handling

Both applications process uploaded source material on the web server.
PostgreSQL stores the resulting safety-processed draft, its audit, and review
status. Uploads are temporary. Optional provider compression sends selected
redacted chunks to the configured model. Users can also copy drafts to their
own Notion database, which must be shared with the Notion integration. Importing
a decision requires the original pack hash. Provider and Notion keys entered by
users are used for the request and are kept out of saved drafts and transactions.

A successful context generation costs exactly €0.499. Wallet amounts and
entries use integer thousandths of a euro. Mollie presets are €5, €10, and €20.
A reserved charge is returned on failure or lease expiry.
