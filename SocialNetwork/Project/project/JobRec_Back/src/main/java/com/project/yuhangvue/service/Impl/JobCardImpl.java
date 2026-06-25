package com.project.yuhangvue.service.Impl;/*
 *   @Author:田宇航
 *   @Date: 2025/4/17 12:17
 */

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.entity.FirmCard;
import com.project.yuhangvue.entity.JobCard;
import com.project.yuhangvue.mapper.FirmCardMapper;
import com.project.yuhangvue.mapper.JobCardMapper;
import com.project.yuhangvue.service.JobCardService;
import jakarta.annotation.Resource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class JobCardImpl extends ServiceImpl<JobCardMapper, JobCard> implements JobCardService {
    private static final Logger logger = LoggerFactory.getLogger(JobCardImpl.class);
    @Resource
    private FirmCardMapper firmCardMapper;

    @Override
    public Result<IPage<JobCard>> page(Integer pageNum, Integer pageSize) {
        // 创建分页对象
        Page<JobCard> page = new Page<>(pageNum, pageSize);
        // 创建查询条件，这里可以根据实际需求添加具体的查询条件
        QueryWrapper<JobCard> queryWrapper = new QueryWrapper<>();
        // 调用父类的 page 方法进行分页查询
        IPage<JobCard> resultPage = super.page(page, queryWrapper);

        // 打印查询结果
        for (JobCard job : resultPage.getRecords()) {
            logger.info("Jobsys: {}", job);
        }

        return Result.success(resultPage);
    }

    @Override
    public Result<JobCard> getInfo(Long id) {
        QueryWrapper<FirmCard> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("firm_id", id);
        Long jobId = firmCardMapper.selectOne(queryWrapper).getJobId();
        JobCard list = super.baseMapper.selectById(jobId);

        return Result.success(list);
    }

    @Override
    public Result<JobCard> list(Integer id) {
        LambdaQueryWrapper<JobCard> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(JobCard::getId, id);
        return Result.success(super.baseMapper.selectOne(queryWrapper));
    }
}

