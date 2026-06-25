package com.project.yuhangvue.controller;
/*
 *   @Author:田宇航
 *   @Date: 2025/4/10 08:40
 */

import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.dto.userInfoDTO;
import com.project.yuhangvue.entity.Candidate;
import com.project.yuhangvue.mapper.CandidateMapper;
import jakarta.annotation.Resource;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Validated
@RestController
@RequestMapping("/candidate")
public class CandidateController {
    @Resource
    private CandidateMapper candidateMapper;

    @Transactional
    @PostMapping("/updateInfo")
    public Result<userInfoDTO> updateInfo(@Validated @RequestBody userInfoDTO userInfoDTO) {
        try {
            Candidate candidate = candidateMapper.selectById(userInfoDTO.getId());
            candidate.setNickname(userInfoDTO.getNickname());
            candidate.setEmail(userInfoDTO.getEmail());
            candidate.setPhone(userInfoDTO.getPhone());
            candidate.setGender(userInfoDTO.getGender());
            candidateMapper.updateById(candidate);
            return Result.success(userInfoDTO);
        }
        catch (Exception e) {
            return Result.error(e.getMessage());
        }
    }

    @PostMapping("/getInfo")
    public Result<Candidate> getInfo(@RequestParam Long userId) {
        try {
            Candidate candidate = candidateMapper.selectById(userId);
            return Result.success(candidate);
        }
        catch (Exception e) {
            return Result.error(e.getMessage());
        }
    }

}

