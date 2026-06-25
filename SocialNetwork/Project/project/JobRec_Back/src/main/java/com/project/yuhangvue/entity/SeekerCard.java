package com.project.yuhangvue.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import java.io.Serializable;

import java.util.Date;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.hibernate.validator.constraints.Length;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
/**
 *
 * @TableName seeker_card
 */
@Data
@TableName(value ="seeker_card")
public class SeekerCard implements Serializable {

    /**
     * ID求职卡片信息
     */
    @NotNull(message="[ID求职卡片信息]不能为空")
    @ApiModelProperty("ID求职卡片信息")
    @JsonSerialize(using = ToStringSerializer.class)
    private Long id;
    /**
     * 求职目标
     */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("求职目标")
    @Length(max= 255,message="编码长度不能超过255")
    private String target;
    /**
     * 工作经验
     */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("工作经验")
    @Length(max= 255,message="编码长度不能超过255")
    private String experience;
    /**
     * 即时可到岗状态
     */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("即时可到岗状态")
    @Length(max= 128,message="编码长度不能超过128")
    private String availability;
    /**
     * 关键技能
     */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("关键技能")
    @Length(max= 255,message="编码长度不能超过255")
    private String skills;
    /**
     * 荣誉标签
     */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("荣誉标签")
    @Length(max= 255,message="编码长度不能超过255")
    private String honor;
    /**
     * 优先级
     */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("优先级")
    @Length(max= 255,message="编码长度不能超过255")
    private String priority;
    /**
     * 简历地址
     */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("简历地址")
    @Length(max= 255,message="编码长度不能超过255")
    private String resume;
    /**
     * 创建时间
     */
    @ApiModelProperty("创建时间")
    @TableField(fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8") //指定格式
    private Date createTime;
    /**
     * 更新时间
     */
    @ApiModelProperty("更新时间")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date updateTime;
    /**
     * 大学院校
     */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("大学院校")
    @Length(max= 128,message="编码长度不能超过128")
    private String university;
    /**
     * 学位
     */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("学位")
    @Length(max= 128,message="编码长度不能超过128")
    private String degree;
    /**
     * 学习时期
     */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("学习时期")
    @Length(max= 128,message="编码长度不能超过128")
    private String period;
    /**
     * 专业
     */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("专业")
    @Length(max= 255,message="编码长度不能超过255")
    private String major;
    /**
     *
     */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("")
    @Length(max= 128,message="编码长度不能超过128")
    private String city;



}
