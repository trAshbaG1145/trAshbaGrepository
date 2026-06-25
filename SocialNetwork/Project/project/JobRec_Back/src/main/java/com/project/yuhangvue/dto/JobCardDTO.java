package com.project.yuhangvue.dto;/*
 *   @Author:田宇航
 *   @Date: 2025/4/17 11:39
 */

import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import io.swagger.annotations.ApiModelProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;
import org.hibernate.validator.constraints.Length;

import java.io.Serializable;

@Data
public class JobCardDTO implements Serializable {

    /**
     *
     */
    @NotNull(message="[]不能为空")
    @ApiModelProperty("")
    @JsonSerialize(using = ToStringSerializer.class)
    private Long id;
    /**
     *
     */
    @NotBlank(message="[]不能为空")
    @Size(max= 200,message="编码长度不能超过200")
    @ApiModelProperty("")
    @Length(max= 200,message="编码长度不能超过200")
    private String jobTitle;
    /**
     *
     */
    @NotBlank(message="[]不能为空")
    @Size(max= 200,message="编码长度不能超过200")
    @ApiModelProperty("")
    @Length(max= 200,message="编码长度不能超过200")
    private String companyName;
    /**
     *
     */
    @Size(max= 100,message="编码长度不能超过100")
    @ApiModelProperty("")
    @Length(max= 100,message="编码长度不能超过100")
    private String industry;
    /**
     *
     */
    @Size(max= 50,message="编码长度不能超过50")
    @ApiModelProperty("")
    @Length(max= 50,message="编码长度不能超过50")
    private String salaryRange;
    /**
     *
     */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("")
    @Length(max= 128,message="编码长度不能超过128")
    private String degree;
    /**
     *
     */
    @Size(max= 128,message="编码长度不能超过128")
    @ApiModelProperty("")
    @Length(max= 128,message="编码长度不能超过128")
    private String experience;
    /**
     *
     */
    @Size(max= 100,message="编码长度不能超过100")
    @ApiModelProperty("")
    @Length(max= 100,message="编码长度不能超过100")
    private String location;
    /**
     *
     */
    @Size(max= -1,message="编码长度不能超过-1")
    @ApiModelProperty("")
    @Length(max= -1,message="编码长度不能超过-1")
    private String description;
    /**
     *
     */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("")
    @Length(max= 255,message="编码长度不能超过255")
    private String skill;
    /**
     *
     */
    @Size(max= 255,message="编码长度不能超过255")
    @ApiModelProperty("")
    @Length(max= 255,message="编码长度不能超过255")
    private String welfare;

}

