package com.project.yuhangvue.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.dto.CandidateDTO;
import com.project.yuhangvue.entity.SeekerCard;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestParam;

@Service
public interface SeekCardService extends IService<SeekerCard> {
    Result<IPage<CandidateDTO>> page(@RequestParam Integer pageNum,
                                     @RequestParam Integer pageSize);

    Result<SeekerCard> getInfo(Long userId);

}
