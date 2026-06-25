<template>
  <div id="app">
    <el-upload
      drag
      class="upload-demo"
      ref="uploadRef"
      style="width: 100%;"
      :action="uploadUrl"
      :auto-upload="false"
      :on-change="handleFileChange"
      :before-upload="beforeUpload"
      accept=".pdf,.docx,.doc,.png,.jpg,.jpeg"
    >
      <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
      <div class="el-upload__text">
        拖拽简历至此或<em>点击上传</em>
      </div>
      <template #tip>
        <div class="el-upload__tip">支持 PDF、DOCX、PNG、JPG 文件</div>
      </template>
    </el-upload>
    <el-row>
      <el-button type="success" @click="isPreviewDrawerOpen = true">简历预览</el-button>
      <el-button type="primary" @click="uploadFile" :loading="loading">简历解析</el-button>
      <el-button type="info" @click="isTextDrawerOpen = true">查看文档</el-button>
    </el-row>

    <el-drawer
      v-model="isTextDrawerOpen"
      title="解析文本"
      :placement="'right'"
      :size="650"
      :with-header="true"
      @close="handleTextDrawerClose"
    >
      <pre class="p-20" style="white-space: pre-wrap; overflow-y: auto; height: 800px;">
        {{ extractedText }}
      </pre>
    </el-drawer>
    <el-drawer
      v-model="isPreviewDrawerOpen"
      title="简历预览"
      :placement="'right'"
      size="50%"
      :with-header="true"
      @close="handlePreviewDrawerClose"
    >
        <embed v-if="previewType === 'pdf'"
               :src="previewUrl"
               type="application/pdf"
               width="100%"
               height="100%"
        />
      <div v-if="previewType === 'image'">
        <img :src="previewUrl" alt="预览图片"/>
      </div>
      <div v-if="previewType === 'doc'">
        <div v-html="docPreviewContent"/>
      </div>
    </el-drawer>
    <div v-if="isLoading">
      <el-progress
        :indeterminate="true"
        :percentage="50"
        status="success"
        :stroke-width="15"
        duration="1"
      >
        <el-button text>分析中</el-button>
      </el-progress>
    </div>
    <div v-if="analysisResult"
         class="analysis-container">
      <div
        v-for="(item, index) in parsedResults"
        :key="index"
        class="markdown-box"
      >
        <div v-html="renderMarkdown(item.content)"></div>
      </div>
    </div>
    <el-empty
      v-else-if="!isLoading"
      description="请上传简历进行分析"
      class="mt-20"
    >
    </el-empty>
  </div>
</template>

<script setup>
import axios from 'axios';
import { ref, computed } from 'vue';
import { ElNotification } from 'element-plus';
import { UploadFilled } from '@element-plus/icons-vue';
import { marked } from 'marked';
import mammoth from 'mammoth';
import service from "@/utils/axios.js";

marked.setOptions({
  breaks: true,
  sanitize: true
});

const isLoading = ref(false);
const uploadProgress = ref(0);
const loading = ref(false);
const uploadRef = ref(null);
const selectedFile = ref(null);
const extractedText = ref('');
const previewUrl = ref('');
const previewType = ref('');
const docPreviewContent = ref('');
const analysisResult = ref('');
const uploadUrl = '/api/upload';
const isTextDrawerOpen = ref(false);
const isPreviewDrawerOpen = ref(false);
const hasResume = ref(false);


const parsedResults = computed(() => {
  if (!analysisResult.value) return [];
  const sections = analysisResult.value.split('####').slice(1);
  return sections.map(section => {
    const [titlePart, ...contentParts] = section.split('\n-');
    return {
      title: titlePart.trim(),
      content: `#### ${section.trim()}`
    };
  });
});

const renderMarkdown = (text) => {
  return marked(text);
};

const handleFileChange = async (file) => {
  selectedFile.value = file.raw;
  const fileType = selectedFile.value.type;
  if (fileType === 'application/pdf') {
    previewType.value = 'pdf';
    previewUrl.value = URL.createObjectURL(selectedFile.value);
  } else if (fileType.startsWith('image/')) {
    previewType.value = 'image';
    previewUrl.value = URL.createObjectURL(selectedFile.value);
  } else if (['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'].includes(fileType)) {
    previewType.value = 'doc';
    try {
      const result = await mammoth.convertToHtml({ arrayBuffer: await selectedFile.value.arrayBuffer() });
      docPreviewContent.value = result.value;
    } catch (error) {
      console.error('文档转换出错:', error);
    }
  }
};

const beforeUpload = () => false;

const uploadFile = async () => {
  if (!selectedFile.value) {
    ElNotification({ title: '警告', message: '请先选择文件', type: 'warning' });
    return;
  }
  isLoading.value = true;
  uploadProgress.value = 0;
  loading.value = true;

  const formData = new FormData();
  formData.append('file', selectedFile.value);

  try {
    const { data } = await axios.post(uploadUrl, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        uploadProgress.value = percentCompleted;
      }
    });

    extractedText.value = data.text;
    analysisResult.value = data.result;
  } catch (error) {
    console.error('上传失败:', error);
    ElNotification({
      title: '错误',
      message: '文件解析失败',
      type: 'error',
      duration: 3000
    });
  } finally {
    isLoading.value = false;
    loading.value = false;
  }
};

const handleTextDrawerClose = () => (isTextDrawerOpen.value = false);
const handlePreviewDrawerClose = () => (isPreviewDrawerOpen.value = false);
</script>

<style scoped>
.analysis-container {
  display: grid;
  gap: 24px;
  margin: 30px 20px 0;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
}

.markdown-box {
  padding: 20px;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  border: 1px solid #ebeef5;
  transition: transform 0.2s;
}

.markdown-box:hover {
  transform: translateY(-3px);
}

.markdown-box :deep(h4) {
  color: #409eff;
  margin: 0 0 15px;
  font-size: 18px;
  border-bottom: 2px solid #eee;
  padding-bottom: 8px;
}

.markdown-box :deep(ul) {
  padding-left: 24px;
  margin: 12px 0;
}

.markdown-box :deep(li) {
  margin-bottom: 10px;
  line-height: 1.7;
  color: #606266;
}

.el-empty {
  max-width: 600px;
  margin: 40px auto;
}

.el-drawer__body {
  padding: 0;
}

.mt-20 {
  margin-top: 20px;
}

.upload-demo {
  margin: 20px auto;
}

.el-spin {
  display: flex;
  justify-content: center;
  margin: 40px auto;
  color: #606266;
}

.el-progress {
  margin: 20px auto;
  max-width: 600px;
}
.show{
  width: 100%;
  height: 100vh;
  overflow: auto;
}
</style>