import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs';
rmSync('dist', { recursive: true, force: true });
mkdirSync('dist', { recursive: true });
cpSync('index.html', 'dist/index.html');
// Custom domain for GitHub Pages (agentmilieu.naninithin.me).
if (existsSync('CNAME')) {
  cpSync('CNAME', 'dist/CNAME');
}
console.log('Built AgentMilieu static landing page → dist/index.html');
