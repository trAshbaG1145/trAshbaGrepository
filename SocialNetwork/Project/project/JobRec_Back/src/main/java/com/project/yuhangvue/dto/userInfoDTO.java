package com.project.yuhangvue.dto;/*
 *   @Author:田宇航
 *   @Date: 2025/4/12 20:39
 */

import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import lombok.Data;

@Data
public class userInfoDTO {

    @JsonSerialize(using = ToStringSerializer.class)
    private Long id;

    private String nickname;
    private String email;
    private String phone;
    private Integer gender;
}

