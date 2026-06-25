package com.project.yuhangvue.controller;/*
 *   @Author:田宇航
 *   @Date: 2025/4/14 08:51
 */

import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.entity.Menu;
import com.project.yuhangvue.service.MenuService;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.annotation.Resource;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

import static com.project.yuhangvue.common.Result.success;

@RestController
@RequestMapping("/menu")
public class MenuController {

    @Resource
    private MenuService menuService;

    @Operation(summary = "获取用户菜单")
    @PostMapping("/getMenu")
    public Result<List<Menu>> getMenu(@RequestParam Long userId) {
        List<Menu> menus = menuService.getMenu(userId);
        return success(menus);
    }

}

