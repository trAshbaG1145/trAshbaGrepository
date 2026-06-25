<template>
  <div id="app">
    <el-row class="row">
      <el-col :span="8">
        <div class="statistic-card">
          <el-statistic :value="98500">
            <template #title>
              <div style="display: inline-flex; align-items: center">
                Daily active users
                <el-tooltip
                    effect="dark"
                    content="Number of users who logged into the product in one day"
                    placement="top"
                >
                  <el-icon style="margin-left: 4px" :size="12">
                    <Warning/>
                  </el-icon>
                </el-tooltip>
              </div>
            </template>
          </el-statistic>
          <div class="statistic-footer">
            <div class="footer-item">
              <span>than yesterday</span>
              <span class="green">
              24%
              <el-icon>
                <CaretTop/>
              </el-icon>
            </span>
            </div>
          </div>
        </div>
      </el-col>
      <el-col :span="8">
        <div class="statistic-card">
          <el-statistic :value="693700">
            <template #title>
              <div style="display: inline-flex; align-items: center">
                Monthly Active Users
                <el-tooltip
                    effect="dark"
                    content="Number of users who logged into the product in one month"
                    placement="top"
                >
                  <el-icon style="margin-left: 4px" :size="12">
                    <Warning/>
                  </el-icon>
                </el-tooltip>
              </div>
            </template>
          </el-statistic>
          <div class="statistic-footer">
            <div class="footer-item">
              <span>month on month</span>
              <span class="red">
              12%
              <el-icon>
                <CaretBottom/>
              </el-icon>
            </span>
            </div>
          </div>
        </div>
      </el-col>
      <el-col :span="8">
        <div class="statistic-card">
          <el-statistic :value="72000" title="New transactions today">
            <template #title>
              <div style="display: inline-flex; align-items: center">
                New transactions today
              </div>
            </template>
          </el-statistic>
          <div class="statistic-footer">
            <div class="footer-item">
              <span>than yesterday</span>
              <span class="green">
              16%
              <el-icon>
                <CaretTop/>
              </el-icon>
            </span>
            </div>
            <div class="footer-item">
              <el-icon :size="14">
                <ArrowRight/>
              </el-icon>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>
    <el-row >
        <div class="log" ref="chartRef"/>
    </el-row>
  </div>
</template>

<script lang="ts" setup>
import {
  ArrowRight,
  CaretBottom,
  CaretTop,
  Warning,
} from '@element-plus/icons-vue'

import * as echarts from 'echarts';
import {onMounted, ref} from "vue";

const chartRef = ref(null);

onMounted(() => {
  const chart = echarts.init(chartRef.value);

  // 从2025年3月1日开始
  const startDate = new Date(2025, 2, 1);
  const endDate = new Date(); // 今天
  const oneDay = 24 * 3600 * 1000;
  const maxLogCount = 150;
  const minLogCount = 50; // 设置一个最小值，避免波动太大

  // 计算日期差（天数）
  const daysDiff = Math.ceil((endDate - startDate) / oneDay);

  // 生成初始数据
  const data = [];
  let prevValue = Math.floor(Math.random() * (maxLogCount - minLogCount)) + minLogCount;

  // 生成从2025年3月1日到今天的日期和日志数量
  for (let i = 0; i <= daysDiff; i++) {
    const currentDate = new Date(startDate.getTime() + i * oneDay);
    const newValue = Math.max(
        minLogCount,
        Math.min(
            maxLogCount,
            Math.round(prevValue * (0.9 + Math.random() * 0.2)) // 波动范围：前一个值的±10%
        )
    );
    data.push([currentDate.getTime(), newValue]);
    prevValue = newValue;
  }

  // 设置图表选项
  const option = {
    tooltip: {
      trigger: 'axis',
      position: function (pt) {
        return [pt[0], '10%'];
      },
      formatter: function (params) {
        const date = new Date(params[0].axisValue);
        return `${date.toLocaleDateString()}<br/>系统日志: ${params[0].value[1]}条`;
      }
    },
    title: {
      left: 'center',
      text: '系统日志数量统计'
    },
    toolbox: {
      feature: {
        dataZoom: {
          yAxisIndex: 'none'
        },
        restore: {},
        saveAsImage: {}
      }
    },
    xAxis: {
      type: 'time',
      boundaryGap: false,
      axisLabel: {
        formatter: function (value) {
          const date = new Date(value);
          return `${date.getMonth() + 1}/${date.getDate()}`;
        }
      }
    },
    yAxis: {
      type: 'value',
      boundaryGap: [0, '100%'],
      max: maxLogCount,
      min: minLogCount,
      name: '日志数量'
    },
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 20
      },
      {
        start: 0,
        end: 20
      }
    ],
    series: [
      {
        name: '系统日志数量',
        type: 'line',
        smooth: true,
        symbol: 'none',
        areaStyle: {
          opacity: 0.3
        },
        lineStyle: {
          width: 2
        },
        data: data
      }
    ]
  };

  chart.setOption(option);

  // 每天自动更新到今天的日期
  function updateChart() {
    const today = new Date();
    const lastDate = new Date(data[data.length - 1][0]);

    // 如果今天已经超过了最后的数据点，添加新数据
    if (today > lastDate) {
      const daysDiff = Math.ceil((today - lastDate) / oneDay);

      for (let i = 1; i <= daysDiff; i++) {
        const currentDate = new Date(lastDate.getTime() + i * oneDay);
        const newValue = Math.max(
            minLogCount,
            Math.min(
                maxLogCount,
                Math.round(prevValue * (0.9 + Math.random() * 0.2))
            )
        );
        data.push([currentDate.getTime(), newValue]);
        prevValue = newValue;
      }

      chart.setOption({
        series: [
          {
            data: data
          }
        ]
      });
    }
  }

  // 初始更新
  updateChart();

  // 每天定时更新
  const updateInterval = setInterval(() => {
    updateChart();

    // 如果今天已经是最后的数据点，停止更新
    const today = new Date();
    const lastDate = new Date(data[data.length - 1][0]);
    if (today <= lastDate) {
      clearInterval(updateInterval);
    }
  }, 24 * 3600 * 1000); // 每24小时更新一次

  // 响应式调整
  window.addEventListener('resize', () => {
    chart.resize();
  });
});
</script>

<style scoped>
#app {
  width: 100%;
  height: 100%;
  padding: 10px;
  display: flex;
  flex-direction: column;
}
:global(h2#card-usage ~ .example .example-showcase) {
  background-color: var(--el-fill-color) !important;
}

.el-statistic {
  --el-statistic-content-font-size: 28px;
}

.statistic-card {
  margin: 5px;
  height: 100%;
  padding: 20px;
  border-radius: 9px;
  background-color: var(--el-bg-color-overlay);
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.statistic-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--el-text-color-regular);
  margin-top: 16px;
}

.statistic-footer .footer-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.statistic-footer .footer-item span:last-child {
  display: inline-flex;
  align-items: center;
  margin-left: 4px;
}

.green {
  color: var(--el-color-success);
}

.red {
  color: var(--el-color-error);
}
.row{
  margin-bottom: 30px;
  width: 99%;
}

.log {
  width: 100%;
  height: 300px;
  margin: 30px auto;
  background-color: white;
  border-radius: 9px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}
</style>
