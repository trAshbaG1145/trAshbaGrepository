package com.project.yuhangvue.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import jakarta.validation.constraints.NotNull;

import java.io.Serializable;

import java.util.Date;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.hibernate.validator.constraints.Length;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
/**
* 
* @TableName role
*/
@Data
@TableName(value ="role")
public class Role implements Serializable {

    /**
    * 
    */
    @NotNull(message="[]不能为空")
    @JsonSerialize(using = ToStringSerializer.class)
    private Long id;
    /**
    * 
    */
    @NotBlank(message="[]不能为空")
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("")
    @Length(max= 255,message="编码长度不能超过255")
    private String role;
    /**
    * 数据权限
    */
    @ApiModelProperty("数据权限")
    private Integer scope;
    /**
    * 角色状态(1-正常 0-停用)
    */
    @ApiModelProperty("角色状态(1-正常 0-停用)")
    private Integer status;
    /**
    * 创建人 ID
    */
    @ApiModelProperty("创建人 ID")
    private Long createBy;
    /**
    * 创建时间
    */
    @ApiModelProperty("创建时间")
    private Date createTime;
    /**
    * 更新人ID
    */
    @ApiModelProperty("更新人ID")
    private Long updateBy;
    /**
    * 更新时间
    */
    @ApiModelProperty("更新时间")
    private Date updateTime;
    /**
    * 逻辑删除标识(0-未删除 1-已删除)
    */
    @NotNull(message="[逻辑删除标识(0-未删除 1-已删除)]不能为空")
    @ApiModelProperty("逻辑删除标识(0-未删除 1-已删除)")
    private Integer isDeleted;
    /**
    * 角色编码
    */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("角色编码")
    @Length(max= 255,message="编码长度不能超过255")
    private String code;


}
