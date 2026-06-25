<template>
  <div class="input-container">
    <el-row>
      <el-col :span="10">
        <el-input v-model="userInput" placeholder="请输入自我简述"></el-input>
      </el-col>
      <el-col :span="3">
        <el-select v-model="experienceInput" placeholder="请选择">
          <el-option label="经验不限" value="经验不限"></el-option>
          <el-option label="1-3年" value="1-3年"></el-option>
          <el-option label="3-5年" value="3-5年"></el-option>
          <el-option label="5年以上" value="5年以上"></el-option>
        </el-select>
      </el-col>
      <!--        <el-col :span="6">-->
      <!--          <el-input v-model="minSalaryInput" placeholder="请输入薪资要求"></el-input>-->
      <!--        </el-col>-->
      <el-col :span="6">
        <el-button
            type="success"
            @click="recommendJobs" :disabled="isLoading">
          {{ isLoading ? '分析中...' : '智能推荐' }}
        </el-button>
      </el-col>
    </el-row>
  </div>
  <div class="result-container">
    <div class="table-container">
      <div v-if="jobs.length>0">
        <h3>{{ message }}</h3>
        <el-table :data="jobs"
                  highlight-current-row
                  style="width: 100%"
                  stripe>
          <el-table-column prop="职位" label="职位" width="200"></el-table-column>
          <el-table-column prop="薪资" label="薪资" width="100"></el-table-column>
          <el-table-column
              min-width="650"
              prop="匹配技能"
              label="技能要求"
              :formatter="(row) => row.匹配技能.join(', ')">
          </el-table-column>
          <el-table-column prop="要求学历" label="学历要求" width="90"></el-table-column>
          <el-table-column prop="经验要求" label="经验要求" width="100"></el-table-column>
          <el-table-column prop="公司" label="公司" width="150"></el-table-column>
          <el-table-column prop="城市" label="城市" width="100"></el-table-column>
          <el-table-column label="时间线">
            <template #default="scope">
              <el-button type="primary"
                         circle
                         @click="handleEdit(scope.$index, scope.row)">
                <el-icon>
                  <DataLine/>
                </el-icon>
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
  <div class="network-container" id="network-container"></div>
</template>

<script setup>
import {ref} from 'vue';
import * as echarts from 'echarts';
import {DataLine} from "@element-plus/icons-vue";

const userInput = ref('');
const experienceInput = ref('');
const minSalaryInput = ref('');
const isLoading = ref(false);
const message = ref('');
const jobs = ref([]);
let myChart = null;

const recommendJobs = async () => {
  isLoading.value = true;
  try {
    const response = await fetch('/api/recommend', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        user_input: userInput.value,
        experience_input: experienceInput.value,
        salary_input: minSalaryInput.value
      })
    });
    const data = await response.json();
    isLoading.value = false;
    updateResults(data);
    if (data.echart_data) {
      console.log(data.echart_data)
      renderECharts(data.echart_data);
    }
  } catch (error) {
    console.error('Error:', error);
    isLoading.value = false;
  } finally {
    isLoading.value = false; // 确保状态重置
  }
};

const updateResults = (data) => {
  message.value = data.message;
  jobs.value = data.jobs;
};

const renderECharts = (data) => {
  const container = document.getElementById('network-container');
  // 配置 ECharts
  const option = {
    title: {
      text: '',
      center: 'center',
    },
    tooltip: {
      trigger: 'item',
      formatter: function (params) {
        return `${params.name}<br/>${params.data.label}`;
      }
    },
    legend: {
      orient: 'vertical',
      left: 'left',
      data: ['职位', '技能', '学历', '经验', '公司', '城市']
    },
    series: [
      {
        type: 'graph',
        layout: 'force',
        symbolSize: 40, // 调整节点的大小
        edgeSymbolSize: [2, 10],
        edgeSymbol: ['circle', 'arrow'],
        data: data.nodes,
        links: data.links,
        categories: [
          {name: '职位', itemStyle: {color: '#4CAF50'}},
          {name: '技能', itemStyle: {color: '#2196F3'}},
          {name: '学历', itemStyle: {color: '#FF9800'}},
          {name: '经验', itemStyle: {color: '#9C27B0'}},
          {name: '公司', itemStyle: {color: '#FF5722'}},
          {name: '城市', itemStyle: {color: '#795548'}}
        ],
        roam: true,
        label: {
          normal: {
            show: true,
            textStyle: {},
          }
        },
        force: {
          repulsion: 2500,
          edgeLength: [10, 50]
        },
        draggable: true,
        labelLayout: {
          hideOverlap: true
        },
        scaleLimit: {
          min: 0.4,
          max: 2
        },
        lineStyle: {
          color: 'source',
        },
        emphasis: {
          focus: 'adjacency',
          label: {
            position: 'right',
            show: true
          }
        },
        edgeLabel: {
          show: true,
          formatter: function (params) {
            return params.data.label || ''; // 如果 label 不存在，返回空字符串
          },
          fontSize: 12,
          color: '#333'
        },
      }
    ]
  };
  // 初始化 ECharts 实例
  if (myChart) {
    myChart.dispose();
  }
  myChart = echarts.init(container);
  myChart.setOption(option);
};

const handleEdit = (index, row) => {
  // 处理编辑逻辑
  console.log(`编辑第 ${index + 1} 行`, row.职位);

}
</script>

<style scoped>

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  margin: 20px;
  background-color: #f5f5f5;
}

.input-container {
  background: white;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
}

input[type="text"] {
  width: 300px;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 5px;
  margin-right: 10px;
}

button {
  padding: 10px 20px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.3s;
}

button:hover {
  background-color: #45a049;
}

.result-container {
  display: flex;
  gap: 20px;
}

.table-container {
  width: 100%;
}

.network-container {
  margin: 0 auto;
  background-color: #ffffff;
  width: 99%;
  height: 420px;
  border-radius: 5px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
</style>
