package com.project.yuhangvue.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.entity.Candidate;
import com.project.yuhangvue.entity.Menu;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.List;

@Service
public interface CandidateService extends IService<Candidate> {
    Result<IPage<Candidate>> page(@RequestParam Integer pageNum,
                                   @RequestParam Integer pageSize);


}
