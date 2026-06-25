<template>
  <el-form ref="form" :model="user" :rules="rules" label-width="80px">
    <el-form-item label="旧密码" prop="oldPassword">
      <el-input
        v-model="user.oldPassword"
        placeholder="请输入旧密码"
        type="password"
        show-password
      />
    </el-form-item>
    <el-form-item label="新密码" prop="newPassword">
      <el-input
        v-model="user.newPassword"
        placeholder="请输入新密码"
        type="password"
        show-password
      />
    </el-form-item>
    <el-form-item label="确认密码" prop="confirmPassword">
      <el-input
        v-model="user.confirmPassword"
        placeholder="请确认新密码"
        type="password"
        show-password
      />
    </el-form-item>
    <el-form-item>
      <el-button type="primary" size="mini" @click="submit">保存</el-button>
      <el-button type="danger" size="mini" @click="close">关闭</el-button>
    </el-form-item>
  </el-form>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import { ElMessage, ElForm } from "element-plus";
import service from "@/utils/axios.js";

const user = ref({
  oldPassword: "",
  newPassword: "",
  confirmPassword: "",
});

// 新增确认密码与新密码一致性验证
const validateConfirmPassword = (rule: any, value: string, callback: any) => {
  if (value !== user.value.newPassword) {
    callback(new Error("两次输入的密码不一致"));
  } else {
    callback();
  }
};

const rules = {
  oldPassword: [
    { required: true, message: "请输入旧密码", trigger: "blur" },
    { min: 6, max: 20, message: "密码长度在 6 到 20 个字符", trigger: "blur" },
  ],
  newPassword: [
    { required: true, message: "请输入新密码", trigger: "blur" },
    { min: 6, max: 20, message: "密码长度在 6 到 20 个字符", trigger: "blur" },
  ],
  confirmPassword: [
    { required: true, message: "请确认新密码", trigger: "blur" },
    { min: 6, max: 20, message: "密码长度在 6 到 20 个字符", trigger: "blur" },
    { validator: validateConfirmPassword, trigger: "blur" }, // 新增自定义验证
  ],
};

const form = ref(null);
const submit = async () => {
  if (!form.value) return;

  // 执行表单验证
  await form.value.validate((valid) => {
    if (!valid) return; // 验证不通过直接返回

    const userId = localStorage.getItem("id"); // 从本地存储获取用户ID
    if (!userId) {
      ElMessage.error("用户信息异常，请重新登录");
      return;
    }
    // 发送密码更新请求
    service
      .post("/user/updatePassword", null, {
        params: {
          // 使用params传递@RequestParam参数
          userId: userId,
          oldPassword: user.value.oldPassword,
          newPassword: user.value.newPassword,
          scope: localStorage.getItem("scope"),
        },
        headers: {
          Authorization: localStorage.getItem("token"), // 假设token存储在localStorage
        },
      })
      .then((response) => {
        if (response.code === 200) {
          ElMessage.success("密码更新成功");
          // 可选：清空表单
          user.value = {
            oldPassword: "",
            newPassword: "",
            confirmPassword: "",
          };
        } else {
          ElMessage.error(response.message || "密码更新失败");
        }
      })
      .catch((error) => {
        ElMessage.error("网络请求失败，请检查网络连接");
        console.error(error);
      });
  });
};

const close = () => {
  // 关闭时重置表单
  user.value = {
    oldPassword: "",
    newPassword: "",
    confirmPassword: "",
  };
  form.value?.resetFields();
};
</script>
