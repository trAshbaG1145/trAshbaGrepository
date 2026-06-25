<template>
  <el-form ref="formRef" :model="form" :rules="rules" label-width="80px">
    <el-form-item label="用户昵称" prop="nickname">
      <el-input v-model="form.nickname" maxlength="30" />
    </el-form-item>
    <el-form-item label="手机号码" prop="phone">
      <el-input v-model="form.phone" maxlength="11" />
    </el-form-item>
    <el-form-item label="邮箱" prop="email">
      <el-input v-model="form.email" maxlength="50" />
    </el-form-item>
    <el-form-item label="性别">
      <el-radio-group v-model="form.gender">
        <el-radio :value="1">男</el-radio>
        <el-radio :value="2">女</el-radio>
        <el-radio :value="0">保密</el-radio>
      </el-radio-group>
    </el-form-item>
    <el-form-item>
      <el-button type="primary" size="mini" @click="submit">保存</el-button>
      <el-button type="danger" size="mini" @click="close">关闭</el-button>
    </el-form-item>
  </el-form>
</template>

<script setup>
import { onMounted, ref } from "vue";
import { ElMessage, ElForm } from "element-plus";
import service from "@/utils/axios.js";

const formRef = ref(null);
const form = ref({
  id: localStorage.getItem("id"),
  nickname: "",
  phone: "",
  email: "",
  gender: null,
});

const rules = {
  nickname: [
    { required: true, message: "请输入用户昵称", trigger: "blur" },
    { min: 3, max: 30, message: "长度在 3 到 30 个字符", trigger: "blur" },
  ],
  phone: [
    { required: true, message: "请输入手机号码", trigger: "blur" },
    {
      pattern: /^1[3456789]\d{9}$/,
      message: "请输入正确的手机号码",
      trigger: "blur",
    },
  ],
  email: [
    { required: true, message: "请输入邮箱", trigger: "blur" },
    { type: "email", message: "请输入正确的邮箱地址", trigger: "blur" },
  ],
};

async function getUserInfo() {
  const userId = localStorage.getItem("id");
  const scope = localStorage.getItem("scope");
  const res = await service.post("/user/getInfo", null, {
    params: { userId: userId, scope: scope },
  });
  if (res.code === 200) {
    form.value = {
      id: res.data.id,
      nickname: res.data.nickname,
      phone: res.data.phone,
      email: res.data.email,
      gender: Number(res.data.gender),
    };
  } else {
    ElMessage.error(res.message);
  }
}

onMounted(() => {
  getUserInfo();
});

const submit = async () => {
  if (formRef.value) {
    await formRef.value.validate((valid) => {
      if (valid) {
        const scope = localStorage.getItem("scope");
        const updateUrl = scope === "2" ? "/firm/updateInfo" : "/candidate/updateInfo";
        service
          .post(updateUrl, form.value)
          .then((response) => {
            if (response.code === 200) {
              ElMessage.success("信息更新成功");
              // 刷新页面
              window.location.reload();
            } else {
              ElMessage.error(response.message);
            }
          })
          .catch((error) => {
            ElMessage.error("请求出错，请稍后重试");
            console.error(error);
          });
      } else {
        ElMessage.error("表单验证失败，请检查输入");
      }
    });
  }
};

const close = () => {
  // 这里可以添加关闭表单的逻辑，比如重置表单数据
  form.value = {
    id: localStorage.getItem("id"),
    nickname: "",
    phone: "",
    email: "",
    gender: "0",
  };
  if (formRef.value) {
    formRef.value.resetFields();
  }
};
</script>

<style scoped></style>
