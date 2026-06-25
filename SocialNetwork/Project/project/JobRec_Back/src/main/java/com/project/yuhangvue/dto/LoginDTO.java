package com.project.yuhangvue.dto;/*
 *   @Author:田宇航
 *   @Date: 2025/4/10 11:36
 */

import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import com.project.yuhangvue.entity.Menu;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.util.List;

@Data
public class LoginDTO {

    @NotBlank(message = "用户名不能为空")
    @Schema( description = "用户名")
    private String username;

    @NotBlank(message = "密码不能为空")
    @Schema( description = "密码")
    private String password;

    @NotBlank(message = "邮箱不能为空")
    @Schema( description = "邮箱")
    private String email;

    @JsonSerialize(using = ToStringSerializer.class)
    private Long id;

    private String token;

    private List<Menu> menu;

    private boolean hasCard;

    private Integer scope;

    public LoginDTO(String token, Long userId, List<Menu> menu, boolean hasCard, Integer scope) {
        this.token = token;
        this.id = userId;
        this.menu = menu;
        this.hasCard = hasCard;
        this.scope = scope;
    }
}

