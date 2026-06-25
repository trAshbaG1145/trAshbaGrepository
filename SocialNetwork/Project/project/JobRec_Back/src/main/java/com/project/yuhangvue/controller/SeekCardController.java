package com.project.yuhangvue.controller;/*
 *   @Author:田宇航
 *   @Date: 2025/4/10 09:35
 */

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.dto.SeekCardDTO;
import com.project.yuhangvue.entity.CandidateCard;
import com.project.yuhangvue.entity.SeekerCard;
import com.project.yuhangvue.mapper.CandidateCardMapper;
import com.project.yuhangvue.mapper.SeekCardMapper;
import com.project.yuhangvue.service.SeekCardService;
import jakarta.annotation.Resource;
import jakarta.validation.constraints.Min;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/seekcard")
public class SeekCardController {

    private static final Logger logger = LoggerFactory.getLogger(SeekCardController.class);

    @Resource
    private SeekCardService seekCardService;

    @Resource
    private SeekCardMapper seekCardMapper;

    @Resource
    private CandidateCardMapper candidateCardMapper;

    @PostMapping("/page")
    public Result page(@RequestParam @Min(value = 1, message = "页码必须大于等于 1") Integer pageNum,
                       @RequestParam @Min(value = 1, message = "每页数量必须大于等于 1") Integer pageSize,
                       @RequestHeader(value = "Accept", defaultValue = "application/json") String acceptHeader) {
        logger.info("Received request with Accept header: {}", acceptHeader);
        try {
            return seekCardService.page(pageNum, pageSize);
        } catch (Exception e) {
            logger.error("分页查询企业信息时出现异常", e);
            return Result.error("分页查询企业信息失败，请稍后重试");
        }
    }

    @Transactional
    @PostMapping("/update")
    public Result update(@RequestBody SeekCardDTO seekCardDTO) {try {
            LambdaQueryWrapper<CandidateCard> queryWrapper = new LambdaQueryWrapper<>();
            queryWrapper.eq(CandidateCard::getCandidateId, seekCardDTO.getUserId());
            CandidateCard existingCard = candidateCardMapper.selectOne(queryWrapper);
            if (existingCard == null) {
                Long id = UUID.randomUUID().getMostSignificantBits();
                var seekerCard = getSeekerCard(seekCardDTO, id);
                seekCardMapper.insert(seekerCard);
                Long userId = seekCardDTO.getUserId();
                CandidateCard candidateCard = new CandidateCard();
                candidateCard.setCandidateId(userId);
                candidateCard.setCardId(id);
                candidateCardMapper.insert(candidateCard);
                return Result.success("求职信息新增成功");
            } else {
                Long id = existingCard.getCardId();
                var seekerCard = getSeekerCard(seekCardDTO, id);
                seekCardMapper.updateById(seekerCard);
                Long userId = seekCardDTO.getUserId();
                CandidateCard candidateCard = new CandidateCard();
                candidateCard.setCandidateId(userId);
                candidateCard.setCardId(id);
                candidateCardMapper.updateById(candidateCard);
                return Result.success("求职信息更新成功");
            }
        } catch (Exception e) {
            logger.error("更新企信息时出现异常", e);
            return Result.error("更新信息失败，请稍后重试");
        }
    }

    @NotNull
    private static SeekerCard getSeekerCard(SeekCardDTO seekCardDTO, Long id) {
        SeekerCard seekerCard = new SeekerCard();
        seekerCard.setId(id);
        seekerCard.setTarget(seekCardDTO.getTarget());
        seekerCard.setExperience(seekCardDTO.getExperience());
        seekerCard.setAvailability(seekCardDTO.getAvailability());
        seekerCard.setSkills(seekCardDTO.getSkills());
        seekerCard.setCity(seekCardDTO.getCity());
        seekerCard.setUniversity(seekCardDTO.getUniversity());
        seekerCard.setMajor(seekCardDTO.getMajor());
        seekerCard.setDegree(seekCardDTO.getDegree());
        seekerCard.setPeriod(seekCardDTO.getPeriod());
        seekerCard.setHonor(seekCardDTO.getHonor());
        seekerCard.setPriority(seekCardDTO.getPriority());
        seekerCard.setResume(seekCardDTO.getResume());
        return seekerCard;
    }

    @PostMapping("/getInfo")
    public Result getInfo(@RequestParam Long userId) {
        return seekCardService.getInfo(userId);
    }

}

