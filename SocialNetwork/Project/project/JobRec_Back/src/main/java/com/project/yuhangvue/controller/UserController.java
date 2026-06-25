package com.project.yuhangvue.controller;

import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.dto.LoginDTO;
import com.project.yuhangvue.service.UserService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RestController
@RequestMapping("/user")
public class UserController {

    @Resource
    private UserService userService;

    @PostMapping("/login")
    public Result login(@RequestBody LoginDTO loginDTO) {
        return userService.login(loginDTO);
    }

    @Transactional
    @PostMapping("/register")
    public Result register(@RequestBody LoginDTO registerDTO) {
        return userService.register(registerDTO);
    }


    @Transactional
    @PostMapping("/updatePassword")
    public Result<String> updatePassword(@RequestParam Long userId, @RequestParam String oldPassword, @RequestParam String newPassword, @RequestParam Integer scope) {
        return userService.updatePassword(userId, oldPassword, newPassword, scope);
    }

    @Transactional
    @PostMapping("/updateAvatar")
    public Result updateAvatar(
            @RequestParam("userId") Long id,  // 匹配前端参数名userId
            @RequestParam Integer scope,
            @RequestParam String avatar) {
       return userService.updateAvatar(id, scope, avatar);
    }

    @PostMapping("/getInfo")
    public Result getInfo(@RequestParam Long userId, @RequestParam Integer scope) {
        return userService.getInfo(userId, scope);
    }

}
