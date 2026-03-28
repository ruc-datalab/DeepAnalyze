#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const rootDir = path.join(__dirname, "..");
const sourceRoot = path.join(rootDir, "node_modules", "monaco-editor");
const targetRoot = path.join(rootDir, "public", "monaco-editor");

function copyDir(src, dest) {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

function copyBundleDir(name) {
  const src = path.join(sourceRoot, name);
  const dest = path.join(targetRoot, name);

  if (!fs.existsSync(src)) {
    throw new Error(`Monaco source directory not found: ${src}`);
  }

  copyDir(src, dest);
  console.log(`Copied Monaco ${name} bundle to public/monaco-editor/${name}`);
}

console.log("Setting up Monaco Editor resources...");

try {
  copyBundleDir("min");
  copyBundleDir("dev");
  console.log("Monaco Editor resources are ready.");
} catch (error) {
  console.error("Error setting up Monaco Editor:", error);
  process.exit(1);
}
