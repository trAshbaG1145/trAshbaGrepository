package com.project.yuhangvue.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.entity.JobCard;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestParam;

@Service
public interface JobCardService extends IService<JobCard> {

    Result<IPage<JobCard>> page(@RequestParam Integer pageNum,
                               @RequestParam Integer pageSize);

    Result<JobCard> getInfo(Long id);


    Result<JobCard> list(Integer id);
}
