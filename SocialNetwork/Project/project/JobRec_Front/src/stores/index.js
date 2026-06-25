// stores/loading.js (修正版)
import { defineStore } from 'pinia'

export const useLoadingStore = defineStore('loading', {
  state: () => ({
    isLoading: false,
    minDuration: 300, // 最小显示时间
    startTime: null   // 新增时间记录字段
  }),
  actions: {
    show() {
      this.startTime = Date.now()  // 记录开始时间
      this.isLoading = true
    },
    hide() {
      this.isLoading = false
    }
  }
})


