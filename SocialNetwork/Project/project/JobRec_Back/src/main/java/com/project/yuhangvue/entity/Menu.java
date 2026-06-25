package com.project.yuhangvue.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import jakarta.validation.constraints.NotNull;
import java.io.Serializable;
import java.util.Date;
import java.util.List;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.hibernate.validator.constraints.Length;

/**
* 
* @TableName menu
*/
@Data
@TableName(value ="menu")
public class Menu implements Serializable {

    /**
    * ID
    */
    @NotNull(message="[ID]不能为空")
    @ApiModelProperty("ID")
    @JsonSerialize(using = ToStringSerializer.class)
    private Long id;
    /**
    * 父菜单ID
    */
    @NotNull(message="[父菜单ID]不能为空")
    @ApiModelProperty("父菜单ID")
    private Long parentId;
    /**
    * 父节点ID路径
    */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("父节点ID路径")
    @Length(max= 255,message="编码长度不能超过255")
    private String treePath;
    /**
    * 菜单名称
    */
    @NotBlank(message="[菜单名称]不能为空")
    @Size(max= 64,message="编码长度不能超过64")
    @ApiModelProperty("菜单名称")
    @Length(max= 64,message="编码长度不能超过64")
    private String name;
    /**
    * 菜单类型（1-菜单 2-目录 3-外链 4-按钮）
    */
    @ApiModelProperty("菜单类型（1-菜单 2-目录 3-外链 4-按钮）")
    private Integer type;
    /**
    * 路由名称（Vue Router 中用于命名路由）
    */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("路由名称（Vue Router 中用于命名路由）")
    @Length(max= 255,message="编码长度不能超过255")
    private String routeName;
    /**
    * 路由路径（Vue Router 中定义的 URL 路径）
    */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("路由路径（Vue Router 中定义的 URL 路径）")
    @Length(max= 128,message="编码长度不能超过128")
    private String routePath;
    /**
    * 组件路径（组件页面完整路径，相对于 src/views/，缺省后缀 .vue）
    */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("组件路径（组件页面完整路径，相对于 src/views/，缺省后缀 .vue）")
    @Length(max= 128,message="编码长度不能超过128")
    private String component;
    /**
    * 【按钮】权限标识
    */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("【按钮】权限标识")
    @Length(max= 128,message="编码长度不能超过128")
    private String perm;
    /**
    * 【目录】只有一个子路由是否始终显示（1-是 0-否）
    */
    @ApiModelProperty("【目录】只有一个子路由是否始终显示（1-是 0-否）")
    private Integer alwaysShow;
    /**
    * 【菜单】是否开启页面缓存（1-是 0-否）
    */
    @ApiModelProperty("【菜单】是否开启页面缓存（1-是 0-否）")
    private Integer keepAlive;
    /**
    * 显示状态（1-显示 0-隐藏）
    */
    @NotNull(message="[显示状态（1-显示 0-隐藏）]不能为空")
    @ApiModelProperty("显示状态（1-显示 0-隐藏）")
    private Integer visible;
    /**
    * 排序
    */
    @ApiModelProperty("排序")
    private Integer sort;
    /**
    * 菜单图标
    */
    @Size(max= 64,message="编码长度不能超过64")
    @ApiModelProperty("菜单图标")
    @Length(max= 64,message="编码长度不能超过64")
    private String icon;
    /**
    * 跳转路径
    */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("跳转路径")
    @Length(max= 128,message="编码长度不能超过128")
    private String redirect;
    /**
    * 创建时间
    */
    @ApiModelProperty("创建时间")
    private Date createTime;
    /**
    * 更新时间
    */
    @ApiModelProperty("更新时间")
    private Date updateTime;
    /**
    * 路由参数
    */
    @ApiModelProperty("路由参数")
    private Object params;

    @TableField(exist = false)
    private List<Menu> children;
}
