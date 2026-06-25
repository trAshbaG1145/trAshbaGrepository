<template>
  <div class="container">
    <div class="job-seeker-cards" v-if="!isLoading">
      <el-card
        v-for="(item, index) in tableData"
        :key="index"
        class="job-seeker-card"
        @click="goToJobDetail(item.id)">
        <div class="job-seeker-header">
          <h2 class="job-seeker-name">{{item.name}}</h2>
          <span class="job-seeker-title">{{ item.target }}</span>
          <p class="job-seeker-contact">
            <i class="icon-email"></i> {{item.email}} |
            <i class="icon-phone"></i> {{item.phone }} |
            <i class="icon-location"></i> {{item.city }}
          </p>
        </div>
        <div class="job-seeker-section">
          <h3><i class="icon-briefcase"></i> 求职意向</h3>
          <p>{{ item.target }} | {{item.experience}} | {{ item.degree }}</p>
        </div>
        <div class="job-seeker-section">
          <h3><i class="icon-graduation-cap"></i> 教育背景</h3>
          <p>{{ item.university }} | {{ item.major }} | {{ item.period }}</p>
        </div>
        <div class="job-seeker-section">
          <h3><i class="icon-honor-cap"></i> 荣誉标签</h3>
          <p>{{item.honor}}</p>
        </div>
        <div class="job-seeker-section">
<!--          <h3><i class="icon-code"></i> 技能</h3>-->
          <div class="tag">
            <el-tag
              v-for="(skill, skillIndex) in item.skills.split('/')"
              :key="skillIndex"
              effect="plain"
              size="large"
              round>
              {{ skill }}
            </el-tag>
          </div>
        </div>

      </el-card>
    </div>
    <el-skeleton v-else-if="isLoading" :rows="5" animated />
    <!-- 底部分页 -->
    <el-pagination
      :current-page="pageNum"
      :page-size="pageSize"
      :total="total"
      :page-sizes="[10, 20, 30]"
      layout="total, sizes, prev, pager, next, jumper"
      @size-change="handlePageSizeChange"
      @current-change="handleCurrentPageChange"/>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { ElPagination, ElCard, ElTag } from 'element-plus';
import { useRouter } from 'vue-router';
import {Medal} from "@element-plus/icons-vue";
import service from "@/utils/axios.js";

// 注册组件
const components = { ElPagination, ElCard, ElTag };

// 响应式数据
const pageNum = ref(1);
const pageSize = ref(10);
const total = ref(0);
const tableData = ref([]);
const router = useRouter();
const isLoading = ref(false);

// 初始化加载数据
onMounted(() => {
  fetchData();
});

// 获取数据函数
const fetchData = async () => {
  isLoading.value = true;
  try {
    const url = '/seekcard/page';
    const response = await service.post(url, null, {
      params: {
        pageNum: pageNum.value,
        pageSize: pageSize.value
      },
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json'
      }
    });

    // 解析分页数据
    const { records, total: totalCount } = response.data;
    tableData.value = records;
    total.value = totalCount;
  } catch (error) {
    console.error('请求出错:', error);
    // 这里可以添加错误提示（如 ElMessage）
  }
  finally {
    isLoading.value = false;
  }
};

// 分页大小变化处理
const handlePageSizeChange = (size) => {
  pageSize.value = size;
  pageNum.value = 1; // 切换页大小后重置为第一页
  fetchData();
};

// 当前页变化处理
const handleCurrentPageChange = (current) => {
  pageNum.value = current;
  fetchData();
};

// 跳转到详情页
const goToJobDetail = (id) => {
  router.push({ name: 'JobDetail', params: { id } });
};
</script>

<style scoped>
.container {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.el-pagination {
  margin: 20px 0;
}

.tag {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: -10px;
}
.job-seeker-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
  gap: 25px;
}

.job-seeker-card {
  background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
  border-radius: 12px;
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.05);
  padding: 10px;
  transition: all 0.3s ease;
  overflow: hidden;
  position: relative;
  border: 1px solid rgba(0, 0, 0, 0.05);
}
.job-seeker-card:hover {
  transform: translateY(-8px);
  transition: box-shadow 0.3s ease, transform 0.3s ease;
  box-shadow: 0 10px 50px 30px rgba(108, 129, 152, 0.18);

}
.job-seeker-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(90deg, #3498db, #2ecc71);
}
.job-seeker-header {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-top: -30px;
}
.job-seeker-name {
  font-size: 1.4rem;
  color: #2c3e50;
  margin-bottom: 5px;
  font-weight: 600;
}
.job-seeker-title {
  font-size: 1.0rem;
  color: #3498db;
  margin-bottom: 10px;
  font-weight: 500;
}
.job-seeker-contact {

  font-size: 0.85rem;
  color: #7f8c8d;
}
.job-seeker-contact i {
  margin-right: 5px;
  color: #3498db;
}
.job-seeker-section {
  margin-bottom: 15px;
}
.job-seeker-section h3 {
  font-size: 1.0rem;
  margin-bottom: 8px;
  color: #2c3e50;
  font-weight: 600;
  display: flex;
  align-items: center;
}
.job-seeker-section h3 i {
  margin-right: 8px;
  font-size: 1.2rem;
}
.job-seeker-section p {
  font-size: 0.9rem;
  line-height: 1.5;
  color: #555;
}
.job-seeker-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}
.job-seeker-btn {
  width: 100%;
  padding: 10px;
  background: linear-gradient(90deg, #3498db, #2980b9);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.95rem;
  transition: all 0.3s ease;
  margin-top: 15px;
  font-weight: 500;
  box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
}
.job-seeker-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.15);
}
/* 图标样式 */
.icon-email::before {
  content: "✉️";
}
.icon-phone::before {
  content: "📞";
}
.icon-location::before {
  content: "📍";
}
.icon-briefcase::before {
  content: "💼";
}
.icon-graduation-cap::before {
  content: "🎓";
}
.icon-honor-cap::before {
  content: "🥇";
}
.icon-code::before {
  content: "💻";
}
</style>
