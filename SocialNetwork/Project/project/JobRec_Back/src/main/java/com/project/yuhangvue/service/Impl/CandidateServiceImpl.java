package com.project.yuhangvue.service.Impl;
/*
 *   @Author:田宇航
 *   @Date: 2025/4/10 09:27
 */

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.entity.Candidate;
import com.project.yuhangvue.mapper.CandidateMapper;
import com.project.yuhangvue.service.CandidateService;
import org.springframework.stereotype.Service;

@Service
public class CandidateServiceImpl extends ServiceImpl<CandidateMapper, Candidate> implements CandidateService {

    @Override
    public Result<IPage<Candidate>> page(Integer pageNum, Integer pageSize) {
        return null;
    }

}

