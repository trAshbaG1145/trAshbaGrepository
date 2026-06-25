package com.project.yuhangvue.entity;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.io.Serializable;
import lombok.Data;

/**
* 
* @TableName candidate_card
*/
@TableName(value ="candidate_card")
@Data
public class CandidateCard implements Serializable {

    /**
    * 
    */
    @TableId
    private Long candidateId;
    /**
    * 
    */
    private Long cardId;

}
