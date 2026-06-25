package com.project.yuhangvue.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import jakarta.validation.constraints.NotNull;
import java.io.Serializable;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

/**
 * 
 * @TableName role_menu
 */
@Data
@TableName(value = "role_menu")
public class RoleMenu implements Serializable {

    /**
     * 角色ID
     */
    @NotNull(message = "[角色ID]不能为空")
    @ApiModelProperty("角色ID")
    private Long roleId;
    /**
     * 菜单ID
     */
    @NotNull(message = "[菜单ID]不能为空")
    @ApiModelProperty("菜单ID")
    private Long menuId;

}
