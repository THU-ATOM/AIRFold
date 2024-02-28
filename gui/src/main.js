import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import VueClipboard from 'vue-clipboard2'

import './main.css';

import 'pdbe-molstar/build/pdbe-molstar-plugin-1.2.1'

VueClipboard.config.autoSetContainer = true;
const app = createApp(App).use(router).use(VueClipboard);

app.mount('#app');
