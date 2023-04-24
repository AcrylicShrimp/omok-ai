import { defineConfig } from 'vite'

export default defineConfig({
    publicDir: './assets',
    // prevent vite from obscuring rust errors
    clearScreen: false,
    // Tauri expects a fixed port, fail if that port is not available
    server: {
        strictPort: true,
    },
    // to make use of `TAURI_PLATFORM`, `TAURI_ARCH`, `TAURI_FAMILY`,
    // `TAURI_PLATFORM_VERSION`, `TAURI_PLATFORM_TYPE` and `TAURI_DEBUG`
    // env variables
    envPrefix: ['VITE_', 'TAURI_'],
    build: {
        // Tauri uses Chromium on Windows and WebKit on macOS and Linux
        target: process.env.TAURI_PLATFORM == 'windows' ? 'chrome105' : 'safari13',
        // don't minify for debug builds
        minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
        // 디버그 빌드에서 소스맵을 제공합니다.
        sourcemap: !!process.env.TAURI_DEBUG,
    },
})