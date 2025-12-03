import { loader } from '@monaco-editor/react';

// 配置Monaco编辑器使用本地资源
export function configureMonaco() {
    if (typeof window !== 'undefined') {
        // 使用本地 node_modules 中的 monaco-editor
        loader.config({
            paths: {
                vs: '/monaco-editor/min/vs'
            }
        });
    }
}
