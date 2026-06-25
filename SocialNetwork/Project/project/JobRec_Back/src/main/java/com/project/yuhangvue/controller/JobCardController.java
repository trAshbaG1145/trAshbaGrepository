package com.project.yuhangvue.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.dto.JobCardDTO;
import com.project.yuhangvue.entity.JobCard;
import com.project.yuhangvue.mapper.FirmCardMapper;
import com.project.yuhangvue.mapper.JobCardMapper;
import com.project.yuhangvue.service.JobCardService;
import com.project.yuhangvue.entity.FirmCard;
import jakarta.annotation.Resource;
import jakarta.validation.constraints.Min;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/job")
@Validated
public class JobCardController {
    private static final Logger logger = LoggerFactory.getLogger(JobCardController.class);

    @Resource
    private JobCardService jobCardService;

    @Resource
    private JobCardMapper jobCardMapper;

    @Resource
    private FirmCardMapper firmCardMapper;


    @PostMapping("/page")
    public Result page(@RequestParam @Min(value = 1, message = "页码必须大于等于 1") Integer pageNum,
                       @RequestParam @Min(value = 1, message = "每页数量必须大于等于 1") Integer pageSize,
                       @RequestHeader(value = "Accept", defaultValue = "application/json") String acceptHeader) {
        logger.info("Received request with Accept header: {}", acceptHeader);
        try {
            return jobCardService.page(pageNum, pageSize);
        } catch (Exception e) {
            logger.error("分页查询企业信息时出现异常", e);
            return Result.error("分页查询企业信息失败，请稍后重试");
        }
    }

    @GetMapping("/detail/{id}")
    public Result<JobCard> getJobDetail(@PathVariable Integer id) {
        return jobCardService.list(id);
    }

    @Transactional
    @PostMapping("/update")
    public Result update(@RequestBody JobCardDTO jobCardDTO) {
        try {
            LambdaQueryWrapper<FirmCard> queryWrapper = new LambdaQueryWrapper<>();
            queryWrapper.eq(FirmCard::getFirmId, jobCardDTO.getId());
            FirmCard existingJobCard = firmCardMapper.selectOne(queryWrapper);
            Long id;
            if (existingJobCard == null) {
                id = UUID.randomUUID().getMostSignificantBits();
                var jobCard = getJobCard(jobCardDTO, id);
                jobCardMapper.insert(jobCard);
                Long jobId = jobCard.getId();
                FirmCard firmCard = new FirmCard();
                firmCard.setFirmId(jobId);
                firmCard.setJobId(id);
                firmCardMapper.insert(firmCard);
                return Result.success("企业信息插入成功");
            } else {
                id = existingJobCard.getJobId();
                var jobCard = getJobCard(jobCardDTO, id);
                jobCardMapper.updateById(jobCard);
                return Result.success("企业信息更新成功");
            }
        } catch (Exception e) {
            logger.error("更新企业信息时出现异常", e);
            return Result.error("更新企业信息失败，请稍后重试");
        }
    }

    private JobCard getJobCard(JobCardDTO jobCardDTO, Long id) {
        JobCard jobCard = new JobCard();
        jobCard.setId(id);
        jobCard.setJobTitle(jobCardDTO.getJobTitle());
        jobCard.setCompanyName(jobCardDTO.getCompanyName());
        jobCard.setIndustry(jobCardDTO.getIndustry());
        jobCard.setSalaryRange(jobCardDTO.getSalaryRange());
        jobCard.setDegree(jobCardDTO.getDegree());
        jobCard.setExperience(jobCardDTO.getExperience());
        jobCard.setLocation(jobCardDTO.getLocation());
        jobCard.setDescription(jobCardDTO.getDescription());
        jobCard.setSkill(jobCardDTO.getSkill());
        jobCard.setWelfare(jobCardDTO.getWelfare());
        return jobCard;
    }

    @PostMapping("/getInfo")
    public Result getInfo(@RequestParam Long userId) {
        return jobCardService.getInfo(userId);
    }
}