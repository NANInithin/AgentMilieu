# AgentMilieu website

The public landing page and authenticated web workspace are served by the same
FastAPI container. The landing page links to the workspace at `/app` and the
public Windows installer at `/downloads/AgentMilieu-Setup.exe`.

Run `npm ci && npm run build` in this directory to build the static landing page.
For a running application and the deployment workflow, see the [deployment guide](../deploy/README.md).
