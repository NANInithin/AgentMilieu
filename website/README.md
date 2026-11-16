# AgentMilieu landing page

Static one-page site for **https://agentmilieu.naninithin.me**, mirroring the
AgentQuantix / AgentVoca `website/` layout.

- `index.html` — the whole page (inline CSS + a tiny script, no dependencies).
- `build.mjs` — copies `index.html` to `dist/index.html`.
- `package.json` / `package-lock.json` — npm `build` script and lockfile for `npm ci`.

Copy this folder into the AgentMilieu repo as `website/`. The Pages pipeline runs
`npm ci` and `npm run build -- --base=/`, then publishes `dist/`.
