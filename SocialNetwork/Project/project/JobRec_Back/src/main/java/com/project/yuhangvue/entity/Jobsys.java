package com.project.yuhangvue.entity;

import lombok.Data;
import jakarta.validation.constraints.Size;
import jakarta.validation.constraints.NotNull;
import io.swagger.annotations.ApiModelProperty;
import jakarta.validation.constraints.NotBlank;
import org.hibernate.validator.constraints.Length;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.io.Serializable;
import java.util.Date;

/**
* 
* @TableName jobSys
 *
*/
@Data
@TableName(value ="jobSys")
public class Jobsys implements Serializable {
    /**
    * 
    */
    @NotNull(message="[]不能为空")
    @ApiModelProperty("")
    @TableId
    private Integer jobId;
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
    private String experience;

    private String degree;

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
    /**
    * 
    */
    @Size(max= 1,message="编码长度不能超过1")
    @ApiModelProperty("")
    @Length(max= 1,message="编码长度不能超过1")
    private String status;
    /**
    * 
    */
    @ApiModelProperty("")
    private Date createdAt;
    /**
    * 
    */
    @ApiModelProperty("")
    private Date updatedAt;

}
