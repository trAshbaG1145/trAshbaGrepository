import axios from 'axios';
import { ElMessage } from 'element-plus';
import router from '../router';


const service = axios.create({
  baseURL: import.meta.env.VITE_APP_BASE_API,
  timeout: 5000
});

service.interceptors.request.use(config => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `${token}`;
  }
  return config;
}, error => {
  console.error('请求错误:', error);
  return Promise.reject(error);
});

service.interceptors.response.use(response => {
  if (response.data.code !== 200) {
    if (response.data.code === 401) {
      ElMessage.error('登录状态失效');
      localStorage.removeItem('token');
      router.push('/login');
    }
    return response.data;
  }
  return response.data;
}, error => {
  if (error.response?.status === 401) {
    localStorage.removeItem('token');
    router.push('/login');
  }
  ElMessage.error(`请求失败: ${error.message}`);
  return Promise.reject(error);
});

export default service;
