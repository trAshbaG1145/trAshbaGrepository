import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import router from './router/index.js'
import "@/assets/base.css"
import "@/assets/main.scss"
import { createPinia } from 'pinia';
import App from './App.vue'
import Particles from "vue3-particles";
import VueCropper from 'vue-cropper';
import 'vue-cropper/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

const app = createApp(App)

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}
app.use(createPinia())
app.use(Particles)
app.use(VueCropper)
app.use(ElementPlus)
app.use(router)
app.mount('#app')
