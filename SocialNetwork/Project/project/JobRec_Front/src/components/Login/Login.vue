<template>
  <div class="user">
    <div class="user_options-container">
      <div class="user_options-text">
        <div class="user_options-unregistered">
          <h2 class="user_unregistered-title">还没有账户?</h2>
          <p class="user_unregistered-text">
            如果您目前还没有账户，不妨考虑注册一个，可以让您能够更好地享受各种服务和功能。</p>
          <button class="user_unregistered-signup" ref="signupButton">注册</button>
        </div>

        <div class="user_options-registered">
          <h2 class="user_registered-title">已有账户了?</h2>
          <p class="user_registered-text">
            如果您已有账户，那么可以直接进行登录操作。登录后，您将能够访问您的个人信息、历史记录以及其他相关功能。</p>
          <button class="user_registered-login" ref="loginButton">登录</button>
        </div>
      </div>

      <div class="user_options-forms" ref="userForms">
        <div class="user_forms-login">
          <h2 class="forms_title">登录</h2>
          <form class="forms_form" @submit.prevent="handleLogin">
            <fieldset class="forms_fieldset">
              <div class="forms_field">
                <input placeholder="用户名" class="forms_field-input" v-model="loginForm.username" required autofocus/>
              </div>
              <div class="forms_field">
                <input type="password" placeholder="密码" class="forms_field-input" v-model="loginForm.password"
                       required/>
              </div>
            </fieldset>
            <div class="forms_buttons">
              <button type="button" class="forms_buttons-forgot">忘记密码了?</button>
              <input type="submit" value="登录" class="forms_buttons-action">
            </div>
          </form>
        </div>
        <div class="user_forms-signup">
          <h2 class="forms_title">注册</h2>
          <form class="forms_form" @submit.prevent="handleRegister">
            <fieldset class="forms_fieldset">
              <div class="forms_field">
                <input type="text" placeholder="用户名" v-model="loginForm.username" class="forms_field-input" required/>
              </div>
              <div class="forms_field">
                <input type="email" placeholder="邮箱" v-model="loginForm.email" class="forms_field-input" required/>
              </div>
              <div class="forms_field">
                <input type="password" placeholder="密码" v-model="loginForm.password" class="forms_field-input" required/>
              </div>
            </fieldset>
            <div class="forms_buttons">
              <input type="submit" value="注册" class="forms_buttons-action">
            </div>
          </form>
        </div>
      </div>
    </div>
<!--    <Particles/>-->
  </div>
</template>

<script setup>
import {ref, onMounted} from 'vue'
import {useRouter} from 'vue-router';
import {setRouters} from "@/router/index.js";
import service from '@/utils/axios.js';
import {ElMessage, ElNotification} from "element-plus";
const router = useRouter(); // 创建路由实例
const signupButton = ref(null)
const loginButton = ref(null)
const userForms = ref(null)

const loginForm = ref({
  username: '',
  password: '',
  email: ''
});
async function handleLogin() {
  // 这里可以添加登录逻辑，例如验证用户名和密码
  const response = await service.post('/user/login', loginForm.value);
  if (response.code === 200) {
    const data = response.data;
    localStorage.setItem('id', data.id)
    localStorage.setItem('token', data.token)
    localStorage.setItem('scope',data.scope)
    localStorage.setItem('menus',JSON.stringify(data.menu));
    // setRouters();
    if (data.hasCard) {
      setRouters();
      await router.push('/index');
      open();
    }else {
      await router.push('/Select');
    }
    // 登录成功，跳转到首页
  } else {
    ElMessage.error(response.message);
  }
};
async function handleRegister() {
  // 这里可以添加注册逻辑，例如验证用户名和密码
  localStorage.setItem('r_username', loginForm.value.username)
  localStorage.setItem('r_password', loginForm.value.password)
  localStorage.setItem('r_email', loginForm.value.email)
  await router.push('/Select');
}
const open = () => {
  ElMessage({
    message: '登录成功',
    type: 'success',
  })
}

onMounted(() => {
  // Add event listeners after component is mounted
  signupButton.value.addEventListener('click', () => {
    userForms.value.classList.remove('bounceRight')
    userForms.value.classList.add('bounceLeft')
  })

  loginButton.value.addEventListener('click', () => {
    userForms.value.classList.remove('bounceLeft')
    userForms.value.classList.add('bounceRight')
  })
})
</script>
<style scoped lang="sass">
/**
 * General variables
 */
$bdrds: 3px

$white: #fff
$black: #000
$gray: #ccc
$salmon: #e8716d
$smoky-black: rgba(#222222, .85)

$ff: 'Montserrat', sans-serif
$ff-body: 12px
$ff-light: 300
$ff-regular: 400
$ff-medium: 500


/**
 * General configs
 */
*
  box-sizing: border-box

body
  font-family: $ff
  font-size: $ff-body
  line-height: 1em

button
  background-color: transparent
  padding: 0
  border: 0
  outline: 0
  cursor: pointer

input
  background-color: transparent
  padding: 0
  border: 0
  outline: 0

  &[type="submit"]
    cursor: pointer

  &::placeholder
    font-size: .85rem
    font-family: $ff
    font-weight: $ff-light
    letter-spacing: .1rem
    color: $gray


/**
 * Bounce to the left side
 */
@keyframes bounceLeft
  0%
    transform: translate3d(100%, -50%, 0)

  50%
    transform: translate3d(-30px, -50%, 0)

  100%
    transform: translate3d(0, -50%, 0)

/**
 * Bounce to the left side
 */
@keyframes bounceRight
  0%
    transform: translate3d(0, -50%, 0)

  50%
    transform: translate3d(calc(100% + 30px), -50%, 0)

  100%
    transform: translate3d(100%, -50%, 0)

/**
 * Show Sign Up form
 */
@keyframes showSignUp
  100%
    opacity: 1
    visibility: visible
    transform: translate3d(0, 0, 0)


/**
 * Page background
 */
.user
  width: 100vw
  height: 100vh
  display: flex
  justify-content: center
  align-items: center
  background: rgb(124, 124, 124)
  background-size: cover

  &_options-container
    position: relative
    width: 80%

  &_options-text
    display: flex
    justify-content: space-between
    width: 100%
    background-color: $smoky-black
    border-radius: $bdrds


/**
 * Registered and Unregistered user box and text
 */
.user_options-registered,
.user_options-unregistered
  width: 50%
  padding: 75px 45px

  color: $white
  font-weight: $ff-light

.user_registered-title,
.user_unregistered-title
  margin-bottom: 15px
  font-size: 1.66rem
  line-height: 1em

.user_unregistered-text,
.user_registered-text
  font-size: .83rem
  line-height: 1.4em

.user_registered-login,
.user_unregistered-signup
  margin-top: 30px
  border: 1px solid $gray
  border-radius: $bdrds
  padding: 10px 30px

  color: $white
  text-transform: uppercase
  line-height: 1em
  letter-spacing: .2rem

  transition: background-color .2s ease-in-out, color .2s ease-in-out

  &:hover
    color: $smoky-black
    background-color: $gray


/**
 * Login and signup forms
 */
.user_options-forms
  position: absolute
  top: 50%
  left: 30px

  width: calc(50% - 30px)
  min-height: 420px
  background-color: $white
  border-radius: $bdrds
  box-shadow: 2px 0 15px rgba($black, .25)
  overflow: hidden

  transform: translate3d(100%, -50%, 0)
  transition: transform .4s ease-in-out

  .user_forms-login
    transition: opacity .4s ease-in-out, visibility .4s ease-in-out

  .forms
    &_title
      margin-bottom: 45px

      font-size: 1.5rem
      font-weight: $ff-medium
      line-height: 1em
      text-transform: uppercase
      color: $salmon
      letter-spacing: .1rem

    &_field
      &:not(:last-of-type)
        margin-bottom: 20px

    &_field-input
      width: 100%
      border-bottom: 1px solid $gray
      padding: 6px 20px 6px 6px

      font-family: $ff
      font-size: 1rem
      font-weight: $ff-light
      letter-spacing: .1rem

      transition: border-color .2s ease-in-out

    &_buttons
      display: flex
      justify-content: space-between
      align-items: center

      margin-top: 35px

      &-forgot
        font-family: $ff
        letter-spacing: .1rem
        color: $gray
        text-decoration: underline

        transition: color .2s ease-in-out

      &-action
        background-color: $salmon
        border-radius: $bdrds
        padding: 10px 35px

        font-size: 1rem
        font-family: $ff
        font-weight: $ff-light
        color: $white
        text-transform: uppercase
        letter-spacing: .1rem

        transition: background-color .2s ease-in-out

  .user_forms-signup,
  .user_forms-login
    position: absolute
    top: 70px
    left: 40px

    width: calc(100% - 80px)

    opacity: 0
    visibility: hidden
    transition: opacity .4s ease-in-out, visibility .4s ease-in-out, transform .5s ease-in-out

  .user_forms-signup
    transform: translate3d(120px, 0, 0)

    .forms_buttons
      justify-content: flex-end

  .user_forms-login
    transform: translate3d(0, 0, 0)
    opacity: 1
    visibility: visible


/**
 * Triggers
 */
.user_options-forms
  &.bounceLeft
    animation: bounceLeft 1s forwards

    .user_forms-signup
      animation: showSignUp 1s forwards

    .user_forms-login
      opacity: 0
      visibility: hidden
      transform: translate3d(-120px, 0, 0)

  &.bounceRight
    animation: bounceRight 1s forwards


/**
 * Responsive 990px
 */
@media screen and (max-width: 990px)
  .user_options-forms
    min-height: 350px

    .forms_buttons
      flex-direction: column

    .user_forms-login
      .forms_buttons-action
        margin-top: 30px

    .user_forms-signup,
    .user_forms-login
      top: 40px

  .user_options-registered,
  .user_options-unregistered
    padding: 50px 45px


</style>
