<template>
  <div id="app">
    <div class="input-container">
      <el-form :model="form" ref="formRef" >
      <el-row class="row">
        <el-col :span="5">
          <el-form-item label="职业名">
          <el-input v-model="form.jobName"></el-input>
        </el-form-item>
        </el-col>
        <el-col :span="5">
          <el-form-item label="技能名">
          <el-input v-model="form.skills"></el-input>
        </el-form-item>
        </el-col>
        <el-col :span="5">
          <el-form-item label="学历">
          <el-select v-model="form.education" placeholder="请选择">
            <el-option label="学历不限" value="学历不限"></el-option>
            <el-option label="本科" value="本科"></el-option>
            <el-option label="硕士" value="硕士"></el-option>
            <el-option label="博士" value="博士"></el-option>
          </el-select>

        </el-form-item>
        </el-col>
        <el-col :span="5">
          <el-form-item label="工作经验年限">
          <el-select v-model="form.experience" placeholder="请选择">
            <el-option label="经验不限" value="经验不限"></el-option>
            <el-option label="1-3年" value="1-3年"></el-option>
            <el-option label="3-5年" value="3-5年"></el-option>
            <el-option label="5年以上" value="5年以上"></el-option>
          </el-select>
        </el-form-item>
        </el-col>
        <el-col :span="3">
          <el-form-item>
          <el-button type="primary" @click="search">搜索</el-button>
        </el-form-item>
        </el-col>
      </el-row>
      </el-form>
    </div>

    <div id="chart"></div>
  </div>
</template>

<script setup>
import {onMounted, ref} from 'vue';
import axios from 'axios';
import * as echarts from 'echarts';

const formRef = ref(null);
const form = ref({
  jobName: '',
  skills: '',
  education: '学历不限',
  experience: '经验不限'
});

const search = async () => {
  const skills = form.value.skills.split(',').map(skill => skill.trim());
  const response = await axios.post('/api/search', {
    job_name: form.value.jobName,
    skills,
    education: form.value.education,
    experience: form.value.experience
  });

  const { echart_data } = response.data;
  const chartDom = document.getElementById('chart');
  const myChart = echarts.init(chartDom);
  const option = {
    tooltip: {},
    legend: {
      data: echart_data.categories.map(category => category.name)
    },
    series: [
      {
        type: 'graph',
        layout: 'force',
        symbolSize: 35, // 调整节点的大小
        edgeSymbol: ['circle', 'arrow'],
        data: echart_data.nodes,
        links: echart_data.links,
        categories: echart_data.categories,
        roam: true,
        label: {
          show: true
        },
        force: {
          repulsion: 500,
          edgeLength: [10, 50]
        },
        scaleLimit: {
          min: 0.4,
          max: 2
        },
        labelLayout: {
          hideOverlap: true
        },
        draggable: true,
      }
    ]
  };
  myChart.setOption(option);
};
onMounted(() => {
  search();
})
</script>

<style scoped>
#app{
  position: relative;
  width: 100%;
}
.row{
  display: flex;
  justify-content: center;
  gap: 10px;
}
#chart{
  width: 90%;
  height: 800px;
  margin: 20px auto;
  background-color: #ffffff;
  border: 1px solid #e0e0e0;
  border-radius: 5px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.input-container {
  background: white;
  padding: 10px;
  border-radius: 10px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.19);
}
</style>