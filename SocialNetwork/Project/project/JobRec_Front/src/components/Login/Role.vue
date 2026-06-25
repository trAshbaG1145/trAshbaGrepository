<template>
  <div id="app">
    <button class="next">
      <span class="button_top" @click="skip()"> 跳过此步 </span>
    </button>
    <ul class="wave-menu">
      <li></li>
      <li></li>
      <li></li>
      <li></li>
      <li></li>
      <li></li>
      <li></li>
      <li></li>
      <li></li>
      <li></li>
    </ul>
    <resumeInfo/>
  </div>
</template>

<script lang="ts" setup>
import {useRouter} from "vue-router";
import resumeInfo from "../Person/resumeInfo.vue"
const router = useRouter();
function skip() {
  router.push('/');
}

</script>

<style scoped>
#app {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 100%;
}

/* From Uiverse.io by JkHuger */
.wave-menu {
  border: 4px solid #545FE5;
  border-radius: 50px;
  width: 200px;
  height: 45px;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0;
  cursor: pointer;
  transition: ease 0.2s;
  margin-bottom: 30px;
  background: #fff;
}

.wave-menu li {
  list-style: none;
  height: 30px;
  width: 4px;
  border-radius: 10px;
  background: #545FE5;
  margin: 0 6px;
  padding: 0;
  animation-name: wave1;
  animation-duration: 0.3s;
  animation-iteration-count: infinite;
  animation-direction: alternate;
  transition: ease 0.2s;
}

.wave-menu:hover > li {
  background: #fff;
}

.wave-menu:hover {
  background: #545FE5;
}

.wave-menu li:nth-child(2) {
  animation-name: wave2;
  animation-delay: 0.2s;
}

.wave-menu li:nth-child(3) {
  animation-name: wave3;
  animation-delay: 0.23s;
  animation-duration: 0.4s;
}

.wave-menu li:nth-child(4) {
  animation-name: wave4;
  animation-delay: 0.1s;
  animation-duration: 0.3s;
}

.wave-menu li:nth-child(5) {
  animation-delay: 0.5s;
}

.wave-menu li:nth-child(6) {
  animation-name: wave2;
  animation-duration: 0.5s;
}

.wave-menu li:nth-child(8) {
  animation-name: wave4;
  animation-delay: 0.4s;
  animation-duration: 0.25s;
}

.wave-menu li:nth-child(9) {
  animation-name: wave3;
  animation-delay: 0.15s;
}

@keyframes wave1 {
  from {
    transform: scaleY(1);
  }

  to {
    transform: scaleY(0.5);
  }
}

@keyframes wave2 {
  from {
    transform: scaleY(0.3);
  }

  to {
    transform: scaleY(0.6);
  }
}

@keyframes wave3 {
  from {
    transform: scaleY(0.6);
  }

  to {
    transform: scaleY(0.8);
  }
}

@keyframes wave4 {
  from {
    transform: scaleY(0.2);
  }

  to {
    transform: scaleY(0.5);
  }
}

.custom-style .el-segmented {
  --el-segmented-item-selected-color: var(--el-text-color-primary);
  --el-segmented-item-selected-bg-color: #ffd100;
  --el-border-radius-base: 10px;
}

/* From Uiverse.io by Voxybuns */
.next {
  /* Variables */
  --button_radius: 0.75em;
  --button_color: #e8e8e8;
  --button_outline_color: #000000;
  position: fixed;
  top: 20px;
  right: 20px;
  font-size: 17px;
  font-weight: bold;
  border: none;
  cursor: pointer;
  border-radius: var(--button_radius);
  background: var(--button_outline_color);
}

.button_top {
  display: block;
  box-sizing: border-box;
  border: 2px solid var(--button_outline_color);
  border-radius: var(--button_radius);
  padding: 0.75em 1.5em;
  background: var(--button_color);
  color: var(--button_outline_color);
  transform: translateY(-0.2em);
  transition: transform 0.1s ease;
}

.next:hover .button_top {
  /* Pull the button upwards when hovered */
  transform: translateY(-0.33em);
}

.next:active .button_top {
  /* Push the button downwards when pressed */
  transform: translateY(0);
}

</style>