package com.project.yuhangvue.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import jakarta.validation.constraints.Size;
import jakarta.validation.constraints.NotNull;

import java.io.Serializable;

import java.util.Date;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.hibernate.validator.constraints.Length;

/**
* 
* @TableName firm
*/
@Data
@TableName(value ="firm")
public class Firm implements Serializable, User,Tokenizable {

    /**
    * 
    */
    @NotNull(message="[]不能为空")
    @ApiModelProperty("")
    private Long id;
    /**
    * 昵称
    */
    @Size(max= 64,message="编码长度不能超过64")
    @ApiModelProperty("昵称")
    @Length(max= 64,message="编码长度不能超过64")
    private String nickname;
    /**
    * 用户名
    */
    @Size(max= 64,message="编码长度不能超过64")
    @ApiModelProperty("用户名")
    @Length(max= 64,message="编码长度不能超过64")
    private String username;
    /**
    * 性别((1-男 2-女 0-保密)
    */
    @ApiModelProperty("性别((1-男 2-女 0-保密)")
    private Integer gender;
    /**
    * 密码
    */
    @Size(max= 100,message="编码长度不能超过100")
    @ApiModelProperty("密码")
    @Length(max= 100,message="编码长度不能超过100")
    private String password;
    /**
    * 用户头像
    */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("用户头像")
    @Length(max= 255,message="编码长度不能超过255")
    private String avatar;
    /**
    * 状态((1-正常 0-禁用)
    */
    @ApiModelProperty("状态((1-正常 0-禁用)")
    private Integer status;
    /**
    * 电话号码
    */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("电话号码")
    @Length(max= 255,message="编码长度不能超过255")
    private String phone;
    /**
    * 用户邮箱
    */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("用户邮箱")
    @Length(max= 128,message="编码长度不能超过128")
    private String email;
    /**
    * 创建时间
    */
    @TableField(fill = FieldFill.INSERT)
    @ApiModelProperty("创建时间")
    private Date createTime;
    /**
    * 创建人ID
    */
    @ApiModelProperty("创建人ID")
    private Long createBy;
    /**
    * 更新时间
    */
    @TableField(fill = FieldFill.INSERT_UPDATE)
    @ApiModelProperty("更新时间")
    private Date updateTime;
    /**
    * 修改人ID
    */
    @ApiModelProperty("修改人ID")
    private Long updateBy;
    /**
    * 逻辑删除标识(0-未删除 1-已删除)
    */
    @ApiModelProperty("逻辑删除标识(0-未删除 1-已删除)")
    private Integer isDeleted;
    /**
    * 姓名
    */
    @Size(max= 64,message="编码长度不能超过64")
    @ApiModelProperty("姓名")
    @Length(max= 64,message="编码长度不能超过64")
    private String name;
}
