import { createRouter, createWebHistory } from 'vue-router'
import BoardView from '../views/BoardView.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: BoardView
  },
  {
    path: '/analyze',
    name: 'analyze',
    component: () => import('../views/AnalyzeView.vue')
  },
  {
    path: '/compare',
    name: 'compare',
    component: () => import('../views/CompareView.vue')
  },
  {
    path: '/about',
    name: 'about',
    component: () => import('../views/AboutView.vue')
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

export default router
