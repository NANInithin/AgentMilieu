import { cpSync, mkdirSync, rmSync } from 'node:fs';

rmSync('dist', { recursive: true, force: true });
mkdirSync('dist', { recursive: true });
const landing = new URL('../src/agentmilieu/product/static/index.html', import.meta.url);
cpSync(landing, 'dist/index.html');
console.log('Built AgentMilieu landing page → dist/index.html');
