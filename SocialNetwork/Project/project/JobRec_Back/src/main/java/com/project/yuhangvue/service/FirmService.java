package com.project.yuhangvue.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.entity.Firm;
import com.project.yuhangvue.entity.Jobsys;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestParam;

@Service
public interface FirmService extends IService<Firm> {

}
