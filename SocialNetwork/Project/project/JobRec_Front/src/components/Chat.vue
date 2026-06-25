<template>
  <div id="app" class="container">
    <div class="chat-container">
      <div class="message-list" ref="messageList">
        <div
          v-for="(message, index) in messages"
          :key="index"
          :class="[
            message.type === 'user' ? 'user-message' : 'ai-message',
            'message'
          ]"
        >
          <div v-html="message.content"></div>
        </div>
      </div>
      <div class="input-container">
        <el-input
          v-model="question"
          placeholder="请输入你的问题"
          clearable
          size="large"
          class="input"
          autosize
        />
        <el-button @click="askQuestion"
                   type="primary"
                   size="large"
                   circle
                   :icon="Search"
                   class="button">
        </el-button>
      </div>
    </div>
    <el-alert
      v-if="error"
      :title="error"
      type="error"
      closable
      @close="error = ''"
      class="alert"
    />
    <p style="position: absolute;bottom: -15px;font-size: 14px;color: #7f8c8d">内容由 AI 生成，请仔细甄别</p>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from "vue";
import MarkdownIt from "markdown-it";
import markdownItKatex from "markdown-it-katex";
import { Search} from "@element-plus/icons-vue";
import {ElNotification} from "element-plus";

const md = new MarkdownIt();
md.use(markdownItKatex);

const question = ref("");
const messages = ref([]);
const error = ref("");
const messageList = ref(null);

const renderedAnswer = computed(() => {
  return md.render(messages.value.length > 0 ? messages.value[messages.value.length - 1].content : "");
});

const warn = () => {
  ElNotification({
    title: '警告⚠️',
    message: '请求内容不能为空',
    type: 'warning',
    showClose: false,
  })
}
const askQuestion = async () => {
  if (!question.value) {
    warn();
    return;
  }
  const userMessage = {
    type: "user",
    content: md.render(question.value)
  };
  messages.value.push(userMessage);
  scrollToBottom();

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question: question.value })
    });
    question.value = ""; // 清空输入框
    if (!response.ok) {
      throw new Error("请求失败");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let answerContent = "";
    messages.value.push({
      type: "ai",
      content: ""
    });
    const aiMessage = messages.value[messages.value.length - 1];

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      const chunk = decoder.decode(value);
      answerContent += chunk;
      aiMessage.content = md.render(answerContent);
      scrollToBottom();
    }
  } catch (err) {
    error.value = "发生未知错误";
    messages.value.pop(); // 移除未成功获取答案的 AI 消息
  }
};

const scrollToBottom = () => {
  if (messageList.value) {
    messageList.value.scrollTop = messageList.value.scrollHeight;
  }
};

const scrollToTop = () => {
  if (messageList.value) {
    messageList.value.scrollTop = 0;
  }
};

onMounted(() => {
  scrollToBottom();
});
</script>

<style scoped>
.container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: calc(100vh - 90px);
  padding: 20px;
  box-sizing: border-box;
}

.chat-container {
  display: flex;
  flex-direction: column;
  width: 90%;
  height: 100%;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  overflow: hidden;
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 30px;
  background-color: #fafafa;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.message {
  max-width: 70%;
  padding: 10px;
  border-radius: 8px;
  word-wrap: break-word;
}

.user-message {
  align-self: flex-end;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #04BE02;
  height: 40px;
}
.user-message div{
  color: rgba(217, 217, 217, 0.8);
  line-height: 10px;
  font-size: 18px;
}

.ai-message {
  align-self: flex-start;
  background-color: #e7e7e7;
}

.input-container {
  padding: 10px;
  display: flex;
  border-top: 1px solid #e4e7ed;
  background-color: #fff;
}

.input {
  flex: 1;
  margin-right: 10px;
}

.alert {
  margin-top: 20px;
}

/* 引入 KaTeX 样式 */
@import "katex/dist/katex.min.css";
</style>