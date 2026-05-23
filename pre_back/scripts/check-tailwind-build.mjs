import { execFileSync } from 'node:child_process';
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';

const projectRoot = path.resolve(import.meta.dirname, '..');

execFileSync('npm', ['run', 'build'], {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: true,
});

const assetsDir = path.join(projectRoot, 'dist', 'assets');
const cssFile = readdirSync(assetsDir).find((file) => file.endsWith('.css'));

if (!cssFile) {
    throw new Error('构建结果中没有生成 CSS 资源，Tailwind 样式未生效。');
}

const css = readFileSync(path.join(assetsDir, cssFile), 'utf8');

if (css.includes('@tailwind utilities;')) {
    throw new Error('Tailwind utilities 仍未展开，说明样式管线没有正确处理工具类。');
}

if (!css.includes('.flex{display:flex}')) {
    throw new Error('构建后的 CSS 中缺少 flex 工具类，页面布局样式不会生效。');
}

console.log(`Tailwind build check passed: ${cssFile}`);
