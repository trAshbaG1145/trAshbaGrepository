package com.project.yuhangvue.service;

import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.dto.LoginDTO;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;

@Service
public interface UserService {
    
    Result login(@RequestBody LoginDTO loginDTO);

    Result getInfo(@RequestParam Long userId, @RequestParam Integer scope);

    Result<String> updatePassword(@RequestParam Long userId, @RequestParam String oldPassword, @RequestParam String newPassword, @RequestParam Integer scope);

    Result register(@RequestBody LoginDTO registerDTO);

    Result updateAvatar(
            @RequestParam("userId") Long id,  // 匹配前端参数名userId
            @RequestParam Integer scope,
            @RequestParam String avatar
    );
}
