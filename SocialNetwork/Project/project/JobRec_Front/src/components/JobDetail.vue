<template>
  <div class="job-detail">
    <div class="job-banner">
      <div class="inner home-inner">
        <div class="job-primary detail-box">
          <!-- 添加条件渲染 -->
          <div v-if="jobDetail" class="info-primary">
            <div class="status">
              <span>招聘中</span>
            </div>
            <div class="name">
              <h1>{{ jobDetail.jobTitle }}</h1>
              <span class="salary">{{ jobDetail.salaryRange }}</span>
            </div>
            <p>

              <span class="desc city"><el-icon><Location/></el-icon>{{ jobDetail.location }}</span>

              <span class="desc time"><el-icon><Suitcase/></el-icon>{{ jobDetail.experience }}</span>

              <span class="desc degree"><el-icon><Promotion/></el-icon>{{ jobDetail.degree }}</span>
            </p>
          </div>
          <!-- 可以添加加载状态提示 -->
          <div v-else>
            正在加载职位详情...
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import {ref, onMounted} from 'vue';
import {useRoute} from 'vue-router';
import {ElCard, ElMessage} from 'element-plus';
import service from "@/utils/axios.js";
import {Location, Promotion, Suitcase} from "@element-plus/icons-vue";

const route = useRoute();
const jobDetail = ref(null);

onMounted(async () => {
  const id = route.params.id;
  try {
    const url = `/job/detail/${id}`;
    const response = await service.get(url);
    if (response.code === 200) {
      jobDetail.value = response.data;
    }
  } catch (error) {
    ElMessage.error('获取职位详情失败');
  }
});
</script>

<style scoped>
.job-detail {
  width: 100%;
  height: 100%;
}

.job-banner {
  width: 100%;
  height: auto;
  background: linear-gradient(90deg, #3b526a 0, #345a6d 100%);
  padding: 18px 0 30px 0;
}

.inner {
  margin: 0 auto;
  width: 1184px;
}

.home-inner {
  max-width: 1184px;
}

.job-primary.detail-box {
  border: none;
  height: auto;
  padding: 0;
  margin: 0;
}

.info-primary {
  position: relative;
  padding-top: 35px;
  width: 680px;
}

.status {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1;
  color: #fff;
  height: 22px;
  line-height: 22px;
}

.name {
  width: 680px;
  padding: 0;
  line-height: 41px;
  height: 41px;
  font-size: 32px;
  display: flex;
  flex-direction: row;
}

.job-banner .name h1 {
  font-size: 28px;
  font-weight: 600;
  color: #fff;
  line-height: 40px;
  max-width: 420px;
  margin: 0 20px 0 0;
}

.salary {
  font-size: 32px;
  font-family: kanzhun-Regular, kanzhun, sans-serif;
  color: #f26d49;
  line-height: 41px;
  height: auto;
  font-weight: 400;
  position: relative;
  top: -2px;
}

p {
  color: #fff;
}

.desc {
  font-size: 16px;
  line-height: 22px;
  margin-right: 20px;
  align-items: center;

}

.el-icon {
  margin-right: 3px;
  width: 18px;
  height: 18px;
  margin-top: 5px;

}
</style>
