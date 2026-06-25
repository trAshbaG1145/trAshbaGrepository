<template>
  <div id="app">
    <el-row
      style="
        width: 100%;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
      "
    >
      <el-col :span="6">
        <el-card class="box-card">
          <template #header>
            <div slot="header" class="clearfix">
              <span>个人信息</span>
            </div>
          </template>
          <div style="text-align: center">
            <!--            <el-avatar :size="120" :src="avatar"/>-->
            <Avatar />
          </div>
          <ul class="list-group list-group-striped">
            <li class="list-group-item">
              <svg-icon icon-class="user" />
              用户名称
              <div class="pull-right">{{ username }}</div>
            </li>
            <li class="list-group-item">
              <svg-icon icon-class="phone" />
              手机号码
              <div class="pull-right">{{ phone }}</div>
            </li>
            <li class="list-group-item">
              <svg-icon icon-class="email" />
              用户邮箱
              <div class="pull-right">{{ email }}</div>
            </li>
            <li class="list-group-item">
              <svg-icon icon-class="tree" />
              认证状态
              <div class="pull-right">已上传</div>
            </li>
            <li class="list-group-item">
              <svg-icon icon-class="peoples" />
              推荐卡片
              <div class="pull-right">已完成</div>
            </li>
            <li class="list-group-item">
              <svg-icon icon-class="date" />
              创建日期
              <div class="pull-right">{{ createTime }}</div>
            </li>
          </ul>
        </el-card>
        <Mouse />
      </el-col>
      <el-col :span="18">
        <el-card>
          <div class="clearfix" style="margin-bottom: 10px">
            <span>资料修改</span>
          </div>
          <el-tabs v-model="activeTab">
            <el-tab-pane label="基本资料" name="userinfo">
              <userInfo />
            </el-tab-pane>
            <el-tab-pane v-if="role == 1" label="求职资料" name="resumeinfo">
              <resumeInfo />
            </el-tab-pane>
            <el-tab-pane v-if="role == 2" label="企业资料" name="resumeinfo">
              <firmInfo />
            </el-tab-pane>
            <el-tab-pane label="修改密码" name="resetPwd">
              <restPwd />
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import userInfo from "./userInfo.vue";
import resumeInfo from "./resumeInfo.vue";
import restPwd from "./restPwd.vue";
import { onMounted, ref } from "vue";
import Mouse from "../Compo/Mouse.vue";
import service from "@/utils/axios.js";
import { ElMessage } from "element-plus";
import firmInfo from "@/components/Person/firmInfo.vue";
import Avatar from "@/components/Person/Avatar.vue";

const activeTab = ref("resumeinfo");
const avatar = localStorage.getItem("avatar");
const username = ref();
const phone = ref();
const email = ref();
const circleUrl = ref();
const createTime = ref();
const role = localStorage.getItem("scope");

async function getUserInfo() {
  const userId = localStorage.getItem("id");
  const scope = localStorage.getItem("scope");
  const res = await service.post("/user/getInfo", null, {
    params: { userId: userId, scope: scope },
  });
  if (res.code === 200) {
    username.value = res.data.nickname;
    phone.value = res.data.phone;
    email.value = res.data.email;
    circleUrl.value = res.data.avatar;
    createTime.value = res.data.createTime;
  } else {
    ElMessage.error(res.message);
  }
}

onMounted(() => {
  getUserInfo();
});
</script>

<style scoped>
#app {
  display: flex;
  width: 100%;
  height: 100%;
}
</style>
