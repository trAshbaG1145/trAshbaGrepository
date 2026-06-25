package com.project.yuhangvue.config;/*
 *   @Author:田宇航
 *   @Date: 2025/4/8 11:20
 */

import com.project.yuhangvue.interceptor.LoginInterceptor;
import jakarta.annotation.Resource;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import java.util.List;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Resource
    private LoginInterceptor loginInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        //注册⾃定义拦截器对象
        registry.addInterceptor(loginInterceptor)
                .addPathPatterns("/**")
                .excludePathPatterns("/error")
                .excludePathPatterns("/user/**")
                .excludePathPatterns("/avatar/**")
                .excludePathPatterns("/resume/**")
                .excludePathPatterns("/user/login")
                .excludePathPatterns("/user/register");
    }

    @Override
    public void configureMessageConverters(List<org.springframework.http.converter.HttpMessageConverter<?>> converters) {
        converters.add(new MappingJackson2HttpMessageConverter());
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/swagger-ui/**")
                .addResourceLocations("classpath:/META-INF/resources/");
        registry.addResourceHandler("/webjars/**")
                .addResourceLocations("classpath:/META-INF/resources/webjars/");
        registry.addResourceHandler("/static/**")
                .addResourceLocations("classpath:/static/");
        registry.addResourceHandler("/resume/**")
                .addResourceLocations("classpath:/static/resume/", "file:uploaded-files/resume/");
        registry.addResourceHandler("/avatar/**")
               .addResourceLocations("classpath:/static/avatar/", "file:uploaded-files/images/uploads/");
    }
}
