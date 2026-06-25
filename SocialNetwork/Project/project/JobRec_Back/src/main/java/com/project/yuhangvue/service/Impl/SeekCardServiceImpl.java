package com.project.yuhangvue.service.Impl;
/*
 *   @Author:田宇航
 *   @Date: 2025/4/10 09:27
 */

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.dto.CandidateDTO;
import com.project.yuhangvue.entity.CandidateCard;
import com.project.yuhangvue.entity.SeekerCard;
import com.project.yuhangvue.mapper.CandidateCardMapper;
import com.project.yuhangvue.mapper.SeekCardMapper;
import com.project.yuhangvue.service.SeekCardService;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class SeekCardServiceImpl extends ServiceImpl<SeekCardMapper, SeekerCard> implements SeekCardService {

    @Resource
    private SeekCardMapper seekCardMapper;

    @Resource
    private CandidateCardMapper candidateCardMapper;

    @Override
    public Result<IPage<CandidateDTO>> page(Integer pageNum, Integer pageSize) {
        // 计算分页偏移量
        int offset = (pageNum - 1) * pageSize;
        // 查询分页数据
        List<CandidateDTO> candidates = seekCardMapper.selectCandidates(pageNum, pageSize, offset);
        // 查询总记录数
        Long total = seekCardMapper.countCandidates();
        // 创建分页对象
        IPage<CandidateDTO> page = new Page<>(pageNum, pageSize, total);

        page.setRecords(candidates);
        // 返回结果
        return Result.success(page);
    }

    @Override
    public Result<SeekerCard> getInfo(Long userId) {
        LambdaQueryWrapper<CandidateCard> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(CandidateCard::getCandidateId,userId);
        if (candidateCardMapper.selectOne(queryWrapper) != null) {
            Long cardId = candidateCardMapper.selectOne(queryWrapper).getCardId();
            LambdaQueryWrapper<SeekerCard> queryWrapper1 = new LambdaQueryWrapper<>();
            queryWrapper1.eq(SeekerCard::getId,cardId);
            SeekerCard seekerCard = seekCardMapper.selectOne(queryWrapper1);
            return Result.success(seekerCard);
        }
        else {
            return Result.error("没有找到该用户的卡片信息");
        }
    }
}