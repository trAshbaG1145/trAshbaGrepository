package com.project.yuhangvue.entity;

import com.baomidou.mybatisplus.annotation.TableName;

import java.io.Serializable;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

/**
* 
* @TableName firm_card
*/
@Data
@TableName(value ="firm_card")
public class FirmCard implements Serializable {
    /**
    * 
    */
    @ApiModelProperty("")
    private Long firmId;
    /**
    * 
    */
    @ApiModelProperty("")
    private Long jobId;


}
