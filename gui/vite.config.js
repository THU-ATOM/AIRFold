import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
const path = require("path");
import Components from 'unplugin-vue-components/vite'
import {
  AntDesignVueResolver,
} from 'unplugin-vue-components/resolvers'
import { visualizer } from "rollup-plugin-visualizer";
import vitePluginImp from 'vite-plugin-imp'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    Components({
      resolvers: [
        AntDesignVueResolver(),
      ]
    }),
    vitePluginImp(),
    visualizer(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})
