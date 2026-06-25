import {createRouter, createWebHistory} from 'vue-router'
import JobDetail from "@/components/JobDetail.vue";
import NotFound from "@/components/Compo/404.vue";
import {useLoadingStore} from "@/stores/index.js";
import Home from "@/views/Home.vue";
import {jwtDecode} from "jwt-decode";

const routes = [
    {
        path: '/',
        name: 'login',
        redirect: '/Login',
    },
    {
        path: '/login',
        name: 'Login',
        component: () => import('../components/Login/Login.vue')
    },
    {
        path: '/Select',
        name: 'Select',
        component: () => import('../components/Login/Select.vue'),
    },
    {
        path: '/Role',
        name: 'Role',
        component: () => import('../components/Login/Role.vue'),
    },
    {
        path: '/Role2',
        name: 'Role2',
        component: () => import('../components/Login/Role2.vue'),
    },
    {
        path: '/detail/:id',
        name: 'JobDetail',
        component: JobDetail,
        meta: {requiresAuth: true},
    },
    {
    path: '/:pathMatch(.*)*', // 使用通配符捕获所有未匹配的路径
    name: '404',
    hidden:true,
    component: NotFound,
  },
]

const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes
})

export const setRouters = () => {
    const storeMenus = localStorage.getItem('menus')
    if (storeMenus) {
        const manageMenus = {path: '/', component: Home, redirect: '/index', children: []}
        const menus = JSON.parse(storeMenus)
        menus.forEach(item => {
            if (item.routePath) {
                let itemMenu = {
                    path: item.routePath,
                    name: item.routeName,
                    component: () => import(/* @vite-ignore */ '../components/' + item.component + '.vue'),
                    meta: {requiresAuth: true},
                }
                manageMenus.children.push(itemMenu)
            } else if (item.children.length) {
                item.children.forEach(item => {
                    if (item.routePath) {
                        let itemMenu = {
                            path: item.routePath,
                            name: item.routeName,
                            component: () => import(/* @vite-ignore */ '../components/' + item.component + '.vue'),
                            meta: {requiresAuth: true},
                        }
                        manageMenus.children.push(itemMenu)
                    }
                })
            }
        })
        router.addRoute(manageMenus)

    }
}

setRouters()


router.beforeEach(async (to, from, next) => {
    const loadingStore = useLoadingStore();
    loadingStore.show(); // 这里会自动记录开始时间

    if (to.matched.some(record => record.meta.requiresAuth)) {
        const token = localStorage.getItem('token');
        // 使用 jwt-decode 解码 JWT
        const decoded = jwtDecode(token);

        if (!token) {
            next('/Login');
        } else {
            if (checkTokenExpired(token)) {
                console.log("token已过期，重新登录");
                next('/Login');
            } else {
                const id = decoded.id;
                const nickname = decoded.nickname;

                // 检查 localStorage 中是否已经有 avatar，如果没有则从 JWT 中解析并存储
                if (!localStorage.getItem('avatar')) {
                    const avatar = decoded.avatar;
                    localStorage.setItem('avatar', avatar);
                }

                // 将 id 和 nickname 存储在 localStorage 中
                localStorage.setItem('id', id);
                localStorage.setItem('nickname', nickname);

                next();
            }
        }
    } else {
        next();
    }
});

function decodeJwtToken(token) {
    try {
        return jwtDecode(token);
    } catch (error) {
        console.error('Error decoding JWT:', error);
        return null;
    }
}

function checkTokenExpired(token) {
    const decodedToken = decodeJwtToken(token);
    const expiration = decodedToken.exp * 1000; // 将过期时间转换为毫秒
    return new Date(expiration) < new Date(); // 如果过期时间小于当前时间，则token已过期
}

router.afterEach(async () => {
    const loadingStore = useLoadingStore()
    const elapsed = Date.now() - loadingStore.startTime
    const remaining = Math.max(loadingStore.minDuration - elapsed, 0)

    setTimeout(() => {
        loadingStore.hide()
    }, remaining)
})
export default router