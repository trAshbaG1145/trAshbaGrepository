<script lang="ts" setup>
import { onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { Tickets, Upload } from "@element-plus/icons-vue";
import type { FormInstance, FormRules, ComponentSize } from "element-plus";
import axios from "axios";
import { useRouter } from "vue-router";
import mammoth from "mammoth";
import service from "@/utils/axios.js";

const isPreviewDrawerOpen = ref(false);
const previewUrl = ref("");
const previewType = ref("");
const docPreviewContent = ref("");
const handlePreviewDrawerClose = () => (isPreviewDrawerOpen.value = false);
// 在 script 中添加以下代码

// 新增简历预览函数
const resumeView = async () => {
  const resumeUrl = ruleForm.resume;
  if (!resumeUrl) {
    ElMessage.warning("请先上传简历文件");
    return;
  }

  previewType.value = "";
  docPreviewContent.value = "";
  previewUrl.value = "";

  // 提取文件扩展名（处理带路径的情况，例如 http://xxx.com/resume.docx）
  const ext = resumeUrl.split(".").pop()?.toLowerCase();

  if (ext === "pdf") {
    previewType.value = "pdf";
    previewUrl.value = resumeUrl;
  } else if (["png", "jpg", "jpeg", "gif", "bmp"].includes(ext || "")) {
    previewType.value = "image";
    previewUrl.value = resumeUrl;
  } else if (["doc", "docx"].includes(ext || "")) {
    previewType.value = "doc";
    try {
      // 获取文档内容并转换为 HTML
      const response = await axios.get(resumeUrl, {
        responseType: "arraybuffer",
      });
      const result = await mammoth.convertToHtml({
        arrayBuffer: response.data,
      });
      docPreviewContent.value = result.value;
    } catch (error) {
      console.error("文档预览失败:", error);
      ElMessage.error("不支持的文档格式或转换失败");
      previewType.value = "";
    }
  } else {
    ElMessage.warning("暂不支持该文件类型预览");
    return;
  }

  isPreviewDrawerOpen.value = true; // 打开预览抽屉
};

// 定义表单规则的接口
interface RuleForm {
  userId: string;
  target: string;
  experience: string;
  city: string;
  university: string;
  degree: string;
  major: string;
  skills: string;
  availability: string;
  period: [string, string];
  resume: string;
  honor: string;
}

// 定义表单大小，使用 ref 来创建响应式引用，并指定类型为 ComponentSize
const formSize = ref<ComponentSize>("default");
// 定义表单实例的引用，类型为 FormInstance
const ruleFormRef = ref<FormInstance>();
// 创建响应式的表单数据对象，类型为 RuleForm
const ruleForm = reactive<RuleForm>({
  userId: localStorage.getItem("id") || "",
  target: "",
  experience: "",
  city: "",
  university: "",
  degree: "",
  major: "",
  skills: "",
  availability: "",
  period: ["", ""],
  resume: "",
  honor: "",
});

// 定义可选的到岗状态数组
const locationOptions = ["一周可到岗", "当月内可到岗", "在职"];

// 定义表单验证规则，类型为 FormRules<RuleForm>
const rules = reactive<FormRules<RuleForm>>({
  target: [
    { required: true, message: "请填写职业期望", trigger: "blur" },
    { min: 2, max: 10, message: "长度应不少于 2 个字符", trigger: "blur" },
  ],
});
// 定义一个 ref 来存储选中的文件
const apiBase = import.meta.env.VITE_APP_BASE_API || "";
const uploadUrl = "/file/upload";
// 处理文件选择事件的函数
// 移除原有的 handleFileChange，改用 http-request
const customUpload = async (options: any) => {
  const { file } = options;
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await service.post(uploadUrl, formData);
    if (response.code === 200) {
      ruleForm.resume = `${apiBase}/resume/${response.data}`;
      ElMessage.success("文件上传成功");
    } else {
      ElMessage.error("上传失败：" + response.message);
    }
  } catch (error: any) {
    ElMessage.error("上传失败：" + error.message);
  }
};

// 定义一个函数，用于将日期范围格式化为特定字符串格式
const formatYearRange = (period: [Date, Date] | null): string | null => {
  if (!period) return null;
  const [startDate, endDate] = period;
  const startYear = startDate.getFullYear();
  const endYear = endDate.getFullYear();
  return `${startYear}-${endYear}`;
};
const router = useRouter(); // 创建路由实例
// 定义一个异步函数，用于处理表单提交操作
const submitForm = async (formEl: FormInstance | undefined) => {
  if (!formEl) return;

  try {
    // 使用 async-await 风格的表单验证
    const valid = await formEl.validate();
    if (valid) {
      const formattedPeriod = formatYearRange(
        ruleForm.period as unknown as [Date, Date]
      );
      if (formattedPeriod) {
        const [start, end] = formattedPeriod.split("-");
        ruleForm.period = [start, end];
      }
      // 等待 axios 请求完成（添加 await）
      const response = await service.post(
        "/seekcard/update",
        ruleForm,
      );
      // 这里 response 已经是解析后的 AxiosResponse 实例，可安全访问 data
      if (response.code === 200) {
        ElMessage.success("提交成功");
        window.location.reload();
      }
    }
  } catch (error: any) {
    // 处理验证失败或请求错误
    if (error instanceof Error) {
      console.log("表单验证错误:", error.message);
    } else {
      console.log("请求错误:", error.response?.data || "网络异常");
    }
  }
};

// 定义一个函数，用于重置表单
const resetForm = (formEl: FormInstance | undefined) => {
  // 如果 formEl 为 null 或者 falsy 值，函数直接返回
  if (!formEl) return;
  // 调用 formEl 的 resetFields 方法重置表单字段
  formEl.resetFields();
};

async function getUserInfo() {
  const userId = localStorage.getItem("id");
  const res = await service.post(
    "/seekcard/getInfo",
    null,
    {
      params: { userId: userId },
    }
  );
  if (res.code === 200 && res.data) {
    ruleForm.target = res.data.target;
    ruleForm.experience = res.data.experience;
    ruleForm.city = res.data.city;
    ruleForm.university = res.data.university;
    ruleForm.degree = res.data.degree;
    ruleForm.major = res.data.major;
    ruleForm.skills = res.data.skills;
    if (res.data.period) {
      // 将 "2021-2025" 拆分成 ["2021", "2025"]，并转换为 Date 对象
      ruleForm.period = res.data.period.split("-").map((year: string) => {
        return new Date(parseInt(year), 0); // 例如：new Date(2021, 0) 表示 2021 年 1 月
      });
    } else {
      // 由于 ruleForm.period 类型为 [string, string]，需要初始化为包含两个空字符串的数组
      ruleForm.period = ["", ""];
    }
    ruleForm.resume = res.data.resume;
    ruleForm.honor = res.data.honor;
    ruleForm.availability = res.data.availability;
  }
}

onMounted(() => {
  getUserInfo();
});
</script>

<template>
  <el-form
    ref="ruleFormRef"
    :model="ruleForm"
    :rules="rules"
    style="min-width: 500px"
    label-width="auto"
    class="demo-ruleForm"
    :size="formSize"
    status-icon
  >
    <el-form-item label="职业期望" prop="target">
      <el-input v-model="ruleForm.target" placeholder="请输入职业期望" />
    </el-form-item>
    <el-form-item label="经验/阅历" prop="experience">
      <el-select v-model="ruleForm.experience" placeholder="选择经验">
        <el-option label="暂无经验" value="经验不限" />
        <el-option label="1-3年经验" value="1-3年经验" />
        <el-option label="3-5年经验" value="3-5年经验" />
        <el-option label="5-10年经验" value="5-10年经验" />
        <el-option label="10年以上经验" value="10年以上经验" />
      </el-select>
    </el-form-item>
    <el-form-item label="现居城市" prop="city">
      <el-input v-model="ruleForm.city" placeholder="请输入现居城市" />
    </el-form-item>
    <el-form-item label="毕业院校" prop="university">
      <el-input v-model="ruleForm.university" placeholder="请输入毕业院校" />
    </el-form-item>
    <el-form-item label="最高学历" prop="degree">
      <el-select v-model="ruleForm.degree" placeholder="选择学历">
        <el-option label="本科" value="本科" />
        <el-option label="硕士" value="硕士" />
        <el-option label="博士" value="博士" />
        <el-option label="其他" value="其他" />
      </el-select>
    </el-form-item>
    <el-form-item label="专业选择" prop="name">
      <el-input v-model="ruleForm.major" placeholder="请输入专业选择" />
    </el-form-item>
    <el-form-item label="就学时间" required>
      <el-date-picker
        v-model="ruleForm.period"
        type="yearrange"
        range-separator="至"
        start-placeholder="入学时间"
        end-placeholder="毕业时间"
      />
    </el-form-item>
    <el-form-item label="擅长技术/技能" prop="skills">
      <el-input v-model="ruleForm.skills" type="textarea" />
    </el-form-item>
    <el-form-item label="获得荣誉" prop="skills">
      <el-input v-model="ruleForm.honor" type="textarea" />
    </el-form-item>
    <el-form-item label="简历上传" prop="resume">
      <el-upload
        class="upload-demo"
        auto-upload="false"
        :action="uploadUrl"
        :http-request="customUpload"
        enctype="multipart/form-data"
        :limit="1"
      >
        <el-button size="default" type="primary">
          <el-icon>
            <Upload />
          </el-icon>
          点击上传
        </el-button>
        <el-button type="success" @click.stop="resumeView()">
          <el-icon>
            <Tickets />
          </el-icon>
          简历预览
        </el-button>
        <template #tip>
          <div class="el-upload__tip">
            支持pdf、word、png/jpg格式的文件，文件大小不超过1MB
          </div>
        </template>
      </el-upload>
    </el-form-item>
    <el-form-item label="目前状态" prop="availability" class="custom-style">
      <el-segmented
        v-model="ruleForm.availability"
        :options="locationOptions"
      />
    </el-form-item>
    <el-form-item>
      <el-button type="primary" @click="submitForm(ruleFormRef)" size="large"
        >提交</el-button
      >
      <el-button @click="resetForm(ruleFormRef)" size="large">重置</el-button>
    </el-form-item>
  </el-form>

  <el-drawer
    v-model="isPreviewDrawerOpen"
    title="简历预览"
    :placement="'right'"
    size="50%"
    :with-header="true"
    @close="handlePreviewDrawerClose"
  >
    <embed
      v-if="previewType === 'pdf'"
      :src="previewUrl"
      type="application/pdf"
      width="100%"
      height="100%"
    />
    <div v-if="previewType === 'image'">
      <img :src="previewUrl" alt="预览图片" />
    </div>
    <div v-if="previewType === 'doc'">
      <div v-html="docPreviewContent" />
    </div>
  </el-drawer>
</template>
<style scoped></style>
