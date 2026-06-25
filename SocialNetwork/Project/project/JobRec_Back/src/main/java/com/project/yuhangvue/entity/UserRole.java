package com.project.yuhangvue.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.io.Serializable;

/**
 * @TableName user_role
 */
@Data
@TableName(value = "user_role")
public class UserRole implements Serializable {

    private Long userId;

    private Long roleId;


}
