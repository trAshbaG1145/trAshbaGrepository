package com.project.yuhangvue.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.project.yuhangvue.entity.Candidate;
import com.project.yuhangvue.entity.Jobsys;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CandidateMapper extends BaseMapper<Candidate> {
}
