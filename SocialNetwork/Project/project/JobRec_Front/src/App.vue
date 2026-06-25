<template>
  <el-config-provider :locale="zhCn">
<!--    <Loading/>-->
    <RouterView />
  </el-config-provider>
</template>

<script>
import zhCn from "element-plus/es/locale/lang/zh-cn";
import {defineComponent} from "vue";
import Loading from "@/components/Compo/Loading.vue";

export default defineComponent({
  components: {Loading},
  computed: {
    zhCn() {
      return zhCn
    }
  }
})

</script>


<style scoped>

</style>