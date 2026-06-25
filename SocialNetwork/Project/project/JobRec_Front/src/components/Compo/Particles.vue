<script setup>
//import { loadFull } from "tsparticles"; // if you are going to use `loadFull`, install the "tsparticles" package too.
import {loadSlim} from "tsparticles-slim"; // if you are going to use `loadSlim`, install the "tsparticles-slim" package too.

const particlesInit = async engine => {
  //await loadFull(engine);
  await loadSlim(engine);
};

const particlesLoaded = async container => {
  console.log("Particles container loaded", container);
};
</script>

<template>
  <div id="app">
    <vue-particles
        id="tsparticles"
        :particlesInit="particlesInit"
        :particlesLoaded="particlesLoaded"
    />

    <vue-particles
        id="tsparticles"
        :particlesInit="particlesInit"
        :particlesLoaded="particlesLoaded"
        :options="{
                    fpsLimit: 120,
                    interactivity: {
                        events: {
                            onClick: {
                                enable: true,
                                mode: 'grab'
                            },
                            onHover: {
                                enable: true,
                                mode: 'attract'

                            },
                            resize: true
                        },
                        modes: {
                            bubble: {
                                distance: 400,
                                duration: 10,
                                opacity: 0.8,
                                size: 30
                            },
                            repulse: {
                                distance: 200,
                                duration: 0.4
                            },
                            grab: {
                                distance: 100,
                            },
                            attract: {     // 新增关键配置！！！
                            distance: 50,     // 作用范围
                            duration: 0.8,
                            speed: 1,        // 吸引速度
                            factor: 1        // 吸引力强度
                        }
                        }
                    },
                    particles: {
                        color: {
                            value: '#ffffff'
                        },
                        links: {
                            color: '#ffffff',
                            distance: 150,
                            enable: true,
                            opacity: 0.5,
                            width: 1
                        },
                        collisions: {
                            enable: true
                        },
                        move: {
                            direction: 'none',
                            enable: true,
                            outMode: 'bounce',
                            random: false,
                            speed: 1,
                            straight: false
                        },
                        number: {
                            density: {
                                enable: true,
                                area: 800
                            },
                            value: 100
                        },
                        opacity: {
                            value: 0.5
                        },
                        shape: {
                            type: 'circle'
                        },
                        size: {
                            random: true,
                            value: 3
                        }
                    },
                    detectRetina: true
                }"
    />
  </div>
</template>

<style scoped>
#app {
  z-index: 0;

}

</style>