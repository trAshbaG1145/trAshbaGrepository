package com.project.yuhangvue.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.project.yuhangvue.dto.CandidateDTO;
import com.project.yuhangvue.entity.SeekerCard;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface SeekCardMapper extends BaseMapper<SeekerCard> {

    /**
     * 查询候选人信息
     * @param pageNum 页码
     * @param pageSize 每页数量
     * @param offset 偏移量
     * @return 候选人信息列表
     */
    @Select("<script>" +
            "SELECT " +
            "c.name, " +
            "c.email, " +
            "c.phone, " +
            "s.city, " +
            "s.target, " +
            "s.experience, " +
            "s.availability, " +
            "s.skills, " +
            "s.honor, " +
            "s.university, " +
            "s.major, " +
            "s.degree, " +
            "s.period, " +
            "s.priority " +
            "FROM candidate c " +
            "JOIN candidate_card cc ON c.id = cc.candidate_id " +
            "JOIN seeker_card s ON cc.card_id = s.id " +
            "ORDER BY s.create_time DESC " +
            "LIMIT #{pageSize} OFFSET #{offset}" +
            "</script>")
    List<CandidateDTO> selectCandidates(@Param("pageNum") Integer pageNum,
                                        @Param("pageSize") Integer pageSize,
                                        @Param("offset") Integer offset);

    /**
     * 查询总记录数
     * @return 总记录数
     */
    @Select("<script>" +
            "SELECT COUNT(*) " +
            "FROM candidate c " +
            "JOIN candidate_card cc ON c.id = cc.candidate_id " +
            "JOIN seeker_card s ON cc.card_id = s.id" +
            "</script>")
    Long countCandidates();
}