<template>
  <div class="common-layout">
    <el-container>
      <el-menu
          class="el-menu-vertical-demo"
          :collapse="isCollapse"
          router
          style="height: 100vh"
          :default-openeds="getDefaultOpeneds"
      >
        <el-menu-item class="brand-item">
          <el-icon>
            <Guide/>
          </el-icon>
          <template #title>
            <span class="brand-title">
              <span class="brand-title-main">社交网络数据分析与处理课程项目</span>
              <span class="brand-title-sub">职业关系图谱与推荐系统</span>
            </span>
          </template>
        </el-menu-item>
        <template #title>{{ item.name }}</template>
        <template v-for="item in menus" :key="item.id">
          <!-- 处理带路由的菜单项 -->
          <template v-if="item.routePath">
            <el-menu-item :index="item.routePath">
              <el-icon class="icon">
                <component :is="item.icon"/>
              </el-icon>
              <template #title>{{ item.name }}</template>
            </el-menu-item>
          </template>

          <!-- 处理子菜单（无路由的父级菜单） -->
          <template v-else>
            <!-- 关键修正：将index转为字符串类型 -->
            <el-sub-menu :index="item.id">
              <template #title>
                <el-icon class="icon">
                  <component :is="item.icon"/>
                </el-icon>
                <span>{{ item.name }}</span>
              </template>

              <el-menu-item
                  v-for="child in item.children"
                  :key="child.id"
                  :index="child.routePath"
              >
                <el-icon>
                  <component :is="child.icon"/>
                </el-icon>
                <template #title>{{ child.name }}</template>
              </el-menu-item>
            </el-sub-menu>
          </template>
        </template>
      </el-menu>
      <el-container>
        <el-header class="header">
          <el-radio-group
              class="group"
              v-model="isCollapse">
            <svg v-if="isCollapse"
                 @click="toggleCollapse"
                 xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="25" height="25">
              <path fill="currentColor"
                    d="M128 192h768v128H128zm0 256h512v128H128zm0 256h768v128H128zm576-352 192 160-192 128z"></path>
            </svg>
            <svg v-if="!isCollapse"
                 @click="toggleCollapse"
                 xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="25" height="25">
              <path fill="currentColor"
                    d="M896 192H128v128h768zm0 256H384v128h512zm0 256H128v128h768zM320 384 128 512l192 128z"></path>
            </svg>
          </el-radio-group>
          <!-- From Uiverse.io by vinodjangid07 -->
          <el-tooltip
              class="box-item"
              effect="dark"
              content="退出"
          >
            <el-button class="Btn" @click="loginOut()" type="danger" circle size="large">
              <el-icon size="23">
                <SwitchButton/>
              </el-icon>
            </el-button>
          </el-tooltip>
          <el-tooltip
              class="box-item"
              effect="dark"
              content="全屏"
          >
            <el-button
                class="fullscreen-btn"
                type="primary"
                circle
                size="large"
                @click="toggleFullscreen"
            >
              <el-icon :size="23">
                <CirclePlus/>
              </el-icon>
            </el-button>
          </el-tooltip>

          <div class="avatar">
            <el-avatar
                :src="avatar"
                :size="40"
                style="cursor: pointer"
            />

          </div>
          <div class="bread">
            <el-breadcrumb separator='/'>
              <el-breadcrumb-item :to="{ path: '/' }"></el-breadcrumb-item>
              <el-breadcrumb-item v-for="(item, index) in breadcrumbItems" :key="index">
                {{ item.name }}
              </el-breadcrumb-item>
            </el-breadcrumb>
          </div>
        </el-header>
        <el-main>
          <Loading/>
          <el-scrollbar height="900px">
            <router-view/>
          </el-scrollbar>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>
<script setup>
import {computed, ref, watch} from "vue";
import Loading from "@/components/Compo/Loading.vue";
import {useRoute} from "vue-router";
import {CirclePlus, Guide, HomeFilled} from "@element-plus/icons-vue";

const storeMenus = localStorage.getItem('menus')
const avatar = localStorage.getItem('avatar' || 'https://cube.elemecdn.com/3/7c/3ea6beec64369c2642b92c6726f1epng.png')
const menus = JSON.parse(storeMenus)
const isCollapse = ref(false)

const getDefaultOpeneds = computed(() => {
  return menus.filter(item => !item.routePath).map(item => item.id);
})

function toggleCollapse() {
  isCollapse.value = !isCollapse.value;
}

const route = useRoute();
const breadcrumbItems = ref([]);
// 根据路由路径找到对应的菜单项
const toggleFullscreen = () => {
  const target = document.documentElement;
  if (!document.fullscreenElement) {
    target.requestFullscreen().catch(err => {
      console.error('全屏请求失败:', err);
    });
  } else {
    document.exitFullscreen().catch(err => {
      console.error('退出全屏失败:', err);
    });
  }
};
const findMenuItemByRoutePath = (menu, path) => {
  for (const item of menu) {
    // 完整匹配当前路径或子路径
    if (item.routePath && path.startsWith(`/${item.routePath}`)) {
      return item;
    }
    if (item.children && item.children.length > 0) {
      const found = findMenuItemByRoutePath(item.children, path);
      if (found) return found;
    }
  }
  return null;
};

const findParentItems = (allItems, itemId, result) => {
  const currentItem = findItemById(allItems, itemId);
  if (!currentItem) return;

  result.unshift(currentItem);

  if (currentItem.parentId !== 0) {
    findParentItems(allItems, currentItem.parentId, result);
  }
};

// 辅助函数：通过id查找菜单项
const findItemById = (menu, id) => {
  for (const item of menu) {
    if (item.id === id) return item;
    if (item.children && item.children.length > 0) {
      const found = findItemById(item.children, id);
      if (found) return found;
    }
  }
  return null;
};

const getBreadcrumbItems = (currentPath) => {
  const items = [];
  const currentItem = findMenuItemByRoutePath(menus, currentPath);

  if (!currentItem) return items;

  // 查找所有父级菜单项（包括没有routePath的）
  findParentItems(menus, currentItem.id, items);

  return items; // 返回所有找到的菜单项，不论是否有routePath
};

// 监听路由变化，更新面包屑
watch(
    () => route.path,
    (newPath) => {
      breadcrumbItems.value = getBreadcrumbItems(newPath);
    },
    {immediate: true}
);
const loginOut = () => {
  localStorage.removeItem('id')
  localStorage.removeItem('avatar')
  localStorage.removeItem('token')
  localStorage.removeItem('menus')
  localStorage.removeItem('scope')
  localStorage.removeItem('nickname')
  localStorage.removeItem('username')
  window.location.href = '/login'
}
</script>

<style scoped>
.common-layout {
  height: 100vh;
  width: 100vw;
}

.header {
  position: relative;
  display: flex;
  align-items: center;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.22);
}

.title {
  font-size: 24px;
  font-weight: bold;
}

.el-menu-vertical-demo:not(.el-menu--collapse) {
  width: 260px;
}

.brand-item {
  min-height: 78px;
  height: auto;
  padding-top: 10px;
  padding-bottom: 10px;
}

.brand-title {
  display: flex;
  flex-direction: column;
  gap: 3px;
  line-height: 1.25;
  white-space: normal;
}

.brand-title-main {
  font-size: 12px;
  opacity: 0.82;
}

.brand-title-sub {
  font-size: 15px;
  font-weight: 600;
}

/* From Uiverse.io by vinodjangid07 */
.Btn {
  position: absolute;
  right: 10px;

}

.group {
  position: absolute;
  left: 10px;
}

.bread {
  position: absolute;
  left: 40px;
}

.avatar {
  position: absolute;
  right: 110px;
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 10px;
}

.fullscreen-btn {
  position: absolute;
  right: 60px; /* 调整到退出按钮左侧 */
}
.icon {
  margin-right: 10px;
}
</style>
