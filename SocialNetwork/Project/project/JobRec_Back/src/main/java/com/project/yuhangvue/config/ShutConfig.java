package com.project.yuhangvue.config;/*
 *   @Author:田宇航
 *   @Date: 2025/3/11 08:18
 */

import jakarta.annotation.PreDestroy;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ShutConfig {

    @PreDestroy
    public void onShutdown() {
        // 在这里执行优雅关闭所需的操作
        System.out.println("正在优雅地关闭应用程序...");
    }
}

