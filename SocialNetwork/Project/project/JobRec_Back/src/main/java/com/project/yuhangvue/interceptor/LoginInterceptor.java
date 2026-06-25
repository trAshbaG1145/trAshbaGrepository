package com.project.yuhangvue.interceptor;
/*
 *   @Author:田宇航
 *   @Date: 2025/1/4 20:49
 */

import com.project.yuhangvue.utils.JwtUtil;
import io.jsonwebtoken.Claims;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;
import org.springframework.web.servlet.ModelAndView;
import org.springframework.lang.NonNull;

@Slf4j
@Component
public class LoginInterceptor implements HandlerInterceptor {

    @Override
    // 原 @NotNull 无法解析，替换为 Spring 提供的 @NonNull
    public boolean preHandle(@NonNull HttpServletRequest request, @NonNull HttpServletResponse response,
            @NonNull Object handler) throws Exception {
        log.info("Interceptor已拦截到请求,开始处理");
        long startTime = System.currentTimeMillis();
        request.setAttribute("startTime", startTime);

        if (request.getMethod().toUpperCase().equals("OPTIONS")) {
            return true;
        }
        String path = request.getRequestURI().toString();
        log.info("———路径为{}", path);

        String token = request.getHeader("Authorization");

        if (token == null || token.isEmpty()) {
            // 如果没有Token或Token格式不正确
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.getWriter().write("Unauthorized: Token is missing or malformed");
            log.info("———Token缺失");
            return false;
        }
        // 验证Token
        Claims claims = JwtUtil.verifyJwt(token);
        try {
            if (JwtUtil.validateJwt(token)) { // 验证Token是否有效
                // Token有效，可以继续处理请求
                String nickname = (String) claims.get("nickname");
                log.info("———Token有效,昵称为：{}", nickname);
                return true;
            } else {
                // Token无效
                response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                response.getWriter().write("Unauthorized: Invalid token");
                log.info("———Token无效");
                return false;
            }
        } catch (Exception e) {
            // Token过期或其他异常
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.getWriter().write("Unauthorized: Token has expired or is invalid");
            log.info("———Token过期或无效");
            return false;
        }
    }

    @Override
    public void postHandle(@NonNull HttpServletRequest request, @NonNull HttpServletResponse response,
            @NonNull Object handler,
            ModelAndView modelAndView) throws Exception {
        long startTime = (Long) request.getAttribute("startTime");
        // log.info("请求处理完毕，耗时：{}ms", System.currentTimeMillis() - startTime);
    }

    @Override
    public void afterCompletion(@NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response, @NonNull Object handler, Exception ex) throws Exception {
        log.info("请求处理完毕，开始清理资源");

    }
}
