import { loader } from "@monaco-editor/react";

export function configureMonaco() {
  if (typeof window === "undefined") {
    return;
  }

  // 使用本地 dev 资源，规避 min 资源首次初始化时的一次性异常。
  loader.config({
    paths: {
      vs: "/monaco-editor/dev/vs",
    },
  });
}
