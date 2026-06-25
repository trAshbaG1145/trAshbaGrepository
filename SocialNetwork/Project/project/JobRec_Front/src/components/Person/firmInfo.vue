<script lang="ts" setup>
import { onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import type { FormInstance, FormRules, ComponentSize } from "element-plus";
import { useRouter } from "vue-router";
import service from "@/utils/axios.js";

// 定义表单规则的接口
interface RuleForm {
  id: string;
  jobTitle: string;
  companyName: string;
  industry: string;
  salaryRange: string;
  degree: string;
  skill: string;
  experience: string;
  location: string;
  description: string;
  welfare: string;
}

// 定义表单大小，使用 ref 来创建响应式引用，并指定类型为 ComponentSize
const formSize = ref<ComponentSize>("default");
// 定义表单实例的引用，类型为 FormInstance
const ruleFormRef = ref<FormInstance>();
// 创建响应式的表单数据对象，类型为 RuleForm
const ruleForm = reactive<RuleForm>({
  id: localStorage.getItem("id") || "",
  jobTitle: "",
  companyName: "",
  industry: "",
  salaryRange: "",
  degree: "",
  skill: "",
  experience: "",
  location: "",
  description: "",
  welfare: "",
});

// 定义表单验证规则，类型为 FormRules<RuleForm>
const rules = reactive<FormRules<RuleForm>>({
  jobTitle: [
    { required: true, message: "请填写招聘职业", trigger: "blur" },
    { min: 2, max: 10, message: "长度应不少于 2 个字符", trigger: "blur" },
  ],
  // 这里原代码中 count 字段在 RuleForm 接口中未定义，需要确认是否有误
  degree: [
    {
      required: true,
      message: "Please select Activity count",
      trigger: "change",
    },
  ],
  // 这里原代码中 location 字段在 RuleForm 接口中未定义，需要确认是否有误
  location: [
    {
      required: true,
      message: "Please select a location",
      trigger: "change",
    },
  ],
});

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
      ruleForm.period = formatYearRange(
        ruleForm.period as unknown as [Date, Date]
      );
      // 等待 axios 请求完成（添加 await）
      const response = await service.post(
        "/job/update",
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
  const res = await service.post("/job/getInfo", null, {
    params: { userId: userId },
  });
  if (res.code === 200 && res.data) {
    ruleForm.jobTitle = res.data.jobTitle;
    ruleForm.companyName = res.data.companyName;
    ruleForm.industry = res.data.industry;
    ruleForm.salaryRange = res.data.salaryRange;
    ruleForm.degree = res.data.degree;
    ruleForm.skill = res.data.skill;
    ruleForm.experience = res.data.experience;
    ruleForm.location = res.data.location;
    ruleForm.description = res.data.description;
    ruleForm.welfare = res.data.welfare;
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
    <el-form-item label="招聘职业" prop="jobTitle">
      <el-input v-model="ruleForm.jobTitle" placeholder="请输入招聘职业" />
    </el-form-item>
    <el-form-item label="公司名称" prop="companyName">
      <el-input v-model="ruleForm.companyName" placeholder="请输入公司名称" />
    </el-form-item>
    <el-form-item label="所处城市" prop="location">
      <el-input v-model="ruleForm.location" placeholder="请输入所处城市" />
    </el-form-item>
    <el-form-item label="所属行业" prop="companyName">
      <el-input v-model="ruleForm.industry" placeholder="请输入职业所属行业" />
    </el-form-item>
    <el-form-item label="薪资待遇" prop="salaryRange">
      <el-input
        v-model="ruleForm.salaryRange"
        placeholder="请输入职业薪资水平"
      />
    </el-form-item>
    <el-form-item label="学历要求" prop="degree">
      <el-select v-model="ruleForm.degree" placeholder="选择学历要求">
        <el-option label="本科" value="本科" />
        <el-option label="硕士" value="硕士" />
        <el-option label="博士" value="博士" />
        <el-option label="其他" value="其他" />
      </el-select>
    </el-form-item>
    <el-form-item label="经验/阅历" prop="experience">
      <el-select v-model="ruleForm.experience" placeholder="选择经验要求">
        <el-option label="暂无经验" value="经验不限" />
        <el-option label="1-3年经验" value="1-3年经验" />
        <el-option label="3-5年经验" value="3-5年经验" />
        <el-option label="5-10年经验" value="5-10年经验" />
        <el-option label="10年以上经验" value="10年以上经验" />
      </el-select>
    </el-form-item>
    <el-form-item label="职业简介" prop="description">
      <el-input v-model="ruleForm.description" type="textarea" />
    </el-form-item>
    <el-form-item label="技能要求" prop="skill">
      <el-input v-model="ruleForm.skill" type="textarea" />
    </el-form-item>
    <el-form-item label="福利待遇" prop="welfare">
      <el-input v-model="ruleForm.welfare" type="textarea" />
    </el-form-item>

    <el-form-item>
      <el-button type="primary" @click="submitForm(ruleFormRef)" size="large"
        >提交</el-button
      >
      <el-button @click="resetForm(ruleFormRef)" size="large">重置</el-button>
    </el-form-item>
  </el-form>
</template>
<style scoped></style>
