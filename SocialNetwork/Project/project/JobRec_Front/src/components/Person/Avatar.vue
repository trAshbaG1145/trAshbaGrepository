<template>
  <div id="app">
    <div class="user-info-head" @click="editCropper()">
      <img v-bind:src="avatar" title="点击上传头像" class="img-circle img-lg"/>
    </div>
    <el-dialog title="上传头像" v-model="dialogVisible" width="25%">
      <el-row>
        <el-image
            fit="cover"
            style="width: 120px;height: 120px;"
            :src="previewImage"
        :zoom-rate="1.2"
        :max-scale="7"
        :min-scale="0.2"
        :preview-src-list="[previewImage]"
        show-progress
        />
      </el-row>
      <el-row>
        <span slot="footer" class="dialog-footer">
          <el-upload
              class="upload-demo"
              auto-upload="false"
              :on-change="handleFileChange"
              enctype="multipart/form-data"
              :limit="1"
          >
            <el-button size="default" type="primary">
              <el-icon><Upload/></el-icon> 选择图片
            </el-button>
          <!-- 应用按钮改为触发真正的上传和提交 -->
            <el-button
                type="success"
                @click.stop="handleApply"
              :disabled="!selectedFile"
            >
              <el-icon>
                <Upload/>
              </el-icon>
              应用并上传
          </el-button>
          <template #tip>
              <div class="el-upload__tip">支持png/jpg格式，文件大小不超过5MB</div>
            </template>
          </el-upload>
        </span>
      </el-row>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import {ref} from "vue";
import {Upload} from "@element-plus/icons-vue";
import {ElMessage} from "element-plus";
import service from '@/utils/axios.js';

// 状态管理
const avatar = ref(localStorage.getItem('avatar') || 'https://cube.elemecdn.com/3/7c/3ea6beec64369c2642b92c6726f1epng.png'); // 响应式头像
const dialogVisible = ref(false);
const previewImage = ref<string | null>(null); // 临时预览图
const selectedFile = ref<File | null>(null); // 选中的文件
const apiBase = import.meta.env.VITE_APP_BASE_API || "";
const uploadUrl = '/file/uploadAvatar';

// 选择文件时的处理（仅暂存文件，不立即上传）
const handleFileChange = (file: any) => {
  const newFile = file.raw; // 获取原始File对象
  selectedFile.value = newFile;

  // 生成预览URL
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.value = e.target?.result as string;
  };
  reader.readAsDataURL(newFile);
};

// 点击应用时的真正上传逻辑
const handleApply = async () => {
  if (!selectedFile.value) {
    ElMessage.error('请先选择图片文件');
    return;
  }

  try {
    // 1. 上传文件到文件服务器，获取图片URL
    const formData = new FormData();
    formData.append('file', selectedFile.value);
    const uploadResponse = await service.post(uploadUrl, formData,);

    if (uploadResponse.code !== 200) {
      throw new Error('文件上传失败');
    }

    const avatarUrl = `${apiBase}/avatar/${uploadResponse.data}`;

    // 2. 调用头像更新接口
    const updateResponse = await service.post(
        "/user/updateAvatar",
        null,
        {
          params: {
            userId: localStorage.getItem('id'),
            scope: localStorage.getItem('scope'),
            avatar: avatarUrl
          }
        }
    );

    if (updateResponse.code === 200) {
      // 3. 更新本地存储和视图
      localStorage.setItem('avatar', avatarUrl);
      avatar.value = avatarUrl;
      dialogVisible.value = false;
      ElMessage.success('头像设置成功');

      // 清理临时文件和预览
      selectedFile.value = null;
      previewImage.value = null;
    } else {
      ElMessage.error(`设置失败：${updateResponse.data.msg || '未知错误'}`);
    }

  } catch (error: any) {
    ElMessage.error(`操作失败：${error.message || '网络连接异常'}`);
    // 清理可能的无效临时文件
    selectedFile.value = null;
    previewImage.value = null;
  }
};

// 打开对话框时恢复初始状态
const editCropper = () => {
  dialogVisible.value = true;
  previewImage.value = avatar.value; // 显示当前头像作为初始预览
  selectedFile.value = null; // 清空已选文件
};
</script>


<style scoped>
#app {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;

}

.user-info-head {
  position: relative;
  width: 120px;
  height: 120px;

}

.user-info-head:hover:after {
  content: '+';
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  bottom: 0;
  color: #eee;
  background: rgba(0, 0, 0, 0.5);
  font-size: 24px;
  font-style: normal;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  cursor: pointer;
  line-height: 110px;
  border-radius: 50%;
}

.img-circle {
  border-radius: 50%;
}

.img-lg {
  width: 120px;
  height: 120px;
  object-fit: cover;
}

.el-row {
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;
  margin-bottom: 10px;
}
</style>
