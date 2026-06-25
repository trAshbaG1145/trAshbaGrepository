<script setup>
import {useRouter} from 'vue-router';
import {setRouters} from "@/router/index.js";
import service from "@/utils/axios.js";

const router = useRouter();

const goToRoute = (path) => {
  router.push(path);
};

async function goCandidate() {
  if (localStorage.getItem("r_username") === null || localStorage.getItem("r_password") === null || localStorage.getItem("r_email") === null) {
    await router.push('/');
  } else {
    const res = await service.post("/user/register", {
      "username": localStorage.getItem("r_username"),
      "password": localStorage.getItem("r_password"),
      "email": localStorage.getItem("r_email"),
      "scope": 1
    })
    if (res.code === 200) {
      localStorage.removeItem("r_username");
      localStorage.removeItem("r_password");
      localStorage.removeItem("r_email");
      localStorage.setItem("username", res.data.email);
      localStorage.setItem('id', res.data.id)
      localStorage.setItem('token', res.data.token)
      localStorage.setItem('scope', res.data.scope)
      localStorage.setItem('menus', JSON.stringify(res.data.menu));
      setRouters();
      await router.push('/Role');
    }
  }
}

async function goFirm() {
  if (localStorage.getItem("r_username") === null || localStorage.getItem("r_password") === null || localStorage.getItem("r_email") === null) {
    await router.push('/');
  } else {
    const res = await service.post("/user/register", {
      "username": localStorage.getItem("r_username"),
      "password": localStorage.getItem("r_password"),
      "email": localStorage.getItem("r_email"),
      "scope": 2
    })
    if (res.code === 200) {
      localStorage.removeItem("r_username");
      localStorage.removeItem("r_password");
      localStorage.removeItem("r_email");
      localStorage.setItem("username", res.data.email);
      localStorage.setItem('id', res.data.id)
      localStorage.setItem('token', res.data.token)
      localStorage.setItem('scope', res.data.scope)
      localStorage.setItem('menus', JSON.stringify(res.data.menu));
      setRouters();
      await router.push('/Role2');
    }
  }
}

</script>

<template>
  <div id="app">
    <div class="card" @click="goToRoute('/Role')">
      <div class="card-details">
        <p class="text-title">求职方</p>
        <p class="text-body">Here are the details of the card</p>
      </div>
      <button class="card-button" @click.stop="goCandidate()">More info</button>
    </div>

    <div class="card" @click="goToRoute('/Role2')">
      <div class="card-details">
        <p class="text-title">企业方</p>
        <p class="text-body">Here are the details of the card</p>
      </div>
      <button class="card-button" @click.stop="goFirm()">More info</button>
    </div>
  </div>
</template>

<style scoped>
#app {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: row;
  justify-content: center;
  gap: 150px;
}

/* From Uiverse.io by alexruix */
.card {
  width: 190px;
  height: 254px;
  border-radius: 20px;
  background: #f5f5f5;
  position: relative;
  top: 200px;
  padding: 1.8rem;
  border: 2px solid #c3c6ce;
  transition: 0.5s ease-out;
  overflow: visible;
}

.card-details {
  color: black;
  height: 100%;
  gap: .5em;
  display: grid;
  place-content: center;
}

.card-button {
  transform: translate(-50%, 125%);
  width: 60%;
  border-radius: 1rem;
  border: none;
  background-color: #008bf8;
  color: #fff;
  font-size: 1rem;
  padding: .5rem 1rem;
  position: absolute;
  left: 50%;
  bottom: 0;
  opacity: 0;
  transition: 0.3s ease-out;
}

.text-body {
  color: rgb(134, 134, 134);
}

/*Text*/
.text-title {
  font-size: 1.5em;
  font-weight: bold;
}

/*Hover*/
.card:hover {
  border-color: #008bf8;
  transition: box-shadow .2s ease-in-out;
  box-shadow: 0 10px 50px 30px rgba(108, 129, 152, 0.18);
}

.card:hover .card-button {
  transform: translate(-50%, 50%);
  opacity: 1;
}
</style>
