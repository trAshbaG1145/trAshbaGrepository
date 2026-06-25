<template>
  <div id="app">
    <!-- 输入和分析按钮 -->
    <div class="input-container">
      <el-input
          v-model="careerName"
          placeholder="请输入职业名"
          size="medium"
          clearable
      ></el-input>
      <el-button
          type="primary"
          @click="analyzeCareer"
          :loading="loading"
          size="medium"
      >
        分析职业发展时间线
      </el-button>
    </div>

    <!-- 时间线展示 -->
    <el-container>
      <div v-if="loading" class="loading-container">
        <Loader/> <!-- 自定义加载组件 -->
      </div>
      <el-timeline
          class="timeline-container horizontal-timeline"
          v-if="timelineData.length > 0"
          :reverse="false"
      >
        <el-timeline-item
            v-for="(stage, index) in timelineData"
            :key="index"
        >
          <div class="stage-title">
            <h3>{{ stage.stageName }}</h3>
          </div>

          <div class="stage-content">
            <p class="mb-10">
              <strong>背景：</strong> {{ stage.background }}
            </p>

            <div v-if="stage.keyEvents.length > 0">
              <h4>关键事件：</h4>
              <ul class="list-disc pl-20">
                <li v-for="(event, eventIndex) in stage.keyEvents" :key="eventIndex">
                  {{ event }}
                </li>
              </ul>
            </div>

            <div v-if="stage.skills.length > 0">
              <h4>技能要求：</h4>
              <ul class="list-disc pl-20">
                <li v-for="(skill, skillIndex) in stage.skills" :key="skillIndex">
                  {{ skill }}
                </li>
              </ul>
            </div>
          </div>
        </el-timeline-item>
      </el-timeline>
      <!-- 提示信息 -->
      <el-empty
          v-if="!loading && timelineData.length === 0"
          description="请输入职业名称进行分析"
          class="mt-20"
      >
      </el-empty>
    </el-container>
  </div>
</template>

<script setup>
import { ref, getCurrentInstance } from 'vue';
import Loader from '@/components/Compo/Loader.vue'; // 同步引入组件（也可保持异步加载）

// 响应式数据
const careerName = ref('');
const timelineData = ref([]);
const loading = ref(false);

// 获取 Vue 实例上下文（用于 $message 等方法）
const { proxy } = getCurrentInstance();
const $message = proxy?.$message;

// 分析职业发展时间线
const analyzeCareer = async () => {
  if (!careerName.value.trim()) {
    $message?.warning('请输入有效的职业名称');
    return;
  }

  loading.value = true;
  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ career_name: careerName.value })
    });

    const data = await response.json();
    console.log('API Response:', data);

    if (data.error) {
      $message?.error(data.error);
    } else {
      timelineData.value = parseTimeline(data.timeline);
    }
  } catch (error) {
    $message?.error('请求出错，请稍后再试');
    console.error('请求出错:', error);
  } finally {
    loading.value = false;
  }
};

// 解析时间线数据
const parseTimeline = (rawTimeline) => {
  const timeline = [];
  const stages = rawTimeline.split(/(?=####?\s*\*\*)/g).slice(1);

  stages.forEach(rawStage => {
    const stageHeaderMatch = rawStage.match(/####?\s*\*\*([^**]+)\*\*/i);
    if (!stageHeaderMatch) return;

    const stageName = stageHeaderMatch[1].trim();
    const stageContent = rawStage.slice(stageHeaderMatch[0].length);

    const background = extractSection(stageContent, ['背景']);
    const keyEvents = extractListItems(extractSection(stageContent, ['关键事件']));
    const skills = extractListItems(extractSection(stageContent, ['技术要求', '技能要求']));

    timeline.push({
      stageName,
      background: background.replace(/^[:：]\s*/, ''),
      keyEvents,
      skills
    });
  });

  return timeline;
};

// 提取段落内容
const extractSection = (content, types) => {
  let result = '';
  for (const type of types) {
    if (type === '背景') {
      const regex = new RegExp(`\\*\\*${type}\\*\\*\\s*[:：]?\\s*([^。]+。)`, 'i');
      const match = content.match(regex);
      if (match) {
        result = match[1].trim();
        break;
      }
    } else if (type === '关键事件') {
      const regex = new RegExp(`\\*\\*${type}\\*\\*\\s*[:：]?\\s*([\\s\\S]*?)(?=\\n\\s*-\\s*\\*)`, 'i');
      const match = content.match(regex);
      if (match) {
        result = match[1].trim();
        break;
      }
    } else {
      const regex = new RegExp(`\\*\\*${type}\\*\\*\\s*[:：]?\\s*([\\s\\S]*?)(?=\\n\\s*\\*\\*|$|\\n\\s*-{3})`, 'i');
      const match = content.match(regex);
      if (match) {
        result = match[1].trim();
        break;
      }
    }
  }
  return result;
};

// 提取列表项
const extractListItems = (content) => {
  return content
    .split(/\n/)
    .filter(line => /^\s*[-*•]/.test(line))
    .map(line => line.replace(/^\s*[-*•]\s*/, '').replace(/\*\*/g, '').trim())
    .filter(item => item.length > 0);
};

// 暴露给模板的变量和方法
const components = { Loader }; // 声明组件（如果需要在模板中使用）
</script>

<style scoped>
#app {
  width: 100%;

}

/* 保留之前的样式代码 */
.input-container {
  background: white;
  padding: 10px;
  border-radius: 10px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.19);
}

.timeline-container {
  width: 100%;
}

.stage-title {
  margin-bottom: 15px;
}

.stage-content {
  line-height: 1.6;
}

h3 {
  color: #409eff;
  margin-bottom: 10px;
}

h4 {
  color: #606266;
  margin: 15px 0 10px;
}

.list-disc {
  margin-left: 20px;
}

.mb-10 {
  margin-bottom: 10px;
}

.el-empty {
  max-width: 600px;
  margin: 40px auto;
}
.loading-container {
  max-width: 600px;
  height: 400px; /* 根据加载组件高度调整 */
  margin: 40px auto;
  display: flex;
  justify-content: center;
  align-items: center;
}

</style>