package com.project.yuhangvue.controller;/*
 *   @Author:田宇航
 *   @Date: 2025/4/17 11:30
 */
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.dto.userInfoDTO;
import com.project.yuhangvue.entity.Firm;
import com.project.yuhangvue.mapper.FirmMapper;
import jakarta.annotation.Resource;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Validated
@RestController
@RequestMapping("/firm")
public class FirmController {

    @Resource
    private FirmMapper firmMapper;

    @Transactional
    @PostMapping("/updateInfo")
    public Result<userInfoDTO> updateInfo(@RequestBody userInfoDTO userInfoDTO){
        try {
            Firm firm = firmMapper.selectById(userInfoDTO.getId());
            firm.setNickname(userInfoDTO.getNickname());
            firm.setEmail(userInfoDTO.getEmail());
            firm.setPhone(userInfoDTO.getPhone());
            firm.setGender(userInfoDTO.getGender());
            firmMapper.updateById(firm);
            return Result.success(userInfoDTO);
        }
        catch (Exception e) {
            return Result.error(e.getMessage());
        }
    }

    @PostMapping("/getInfo")
    public Result<Firm> getInfo(@RequestParam Long userId) {
        try {
            Firm firm = firmMapper.selectById(userId);
            return Result.success(firm);
        }
        catch (Exception e) {
            return Result.error(e.getMessage());
        }
    }


}

