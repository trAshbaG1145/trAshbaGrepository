package com.project.yuhangvue.service.Impl;/*
 *   @Author:田宇航
 *   @Date: 2025/4/14 08:53
 */

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.yuhangvue.entity.Menu;
import com.project.yuhangvue.entity.UserRole;
import com.project.yuhangvue.mapper.MenuMapper;
import com.project.yuhangvue.mapper.RMMapper;
import com.project.yuhangvue.mapper.URMapper;
import com.project.yuhangvue.service.MenuService;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class MenuServiceImpl extends ServiceImpl<MenuMapper, Menu> implements MenuService {

    @Resource
    private URMapper urMapper;
    @Resource
    private RMMapper rmMapper;

    @Override
    public List<Menu> getMenu(Long userId) {
        // 根据用户ID查询UserRole，获取角色ID
        LambdaQueryWrapper<UserRole> roleWrapper = new LambdaQueryWrapper<>();
        roleWrapper.eq(UserRole::getUserId, userId);
        UserRole userRole = urMapper.selectOne(roleWrapper);

        List<Long> menuIds = rmMapper.findMenuIdsByRoleId(userRole.getRoleId());

        // List<Menu> menus = menuService.findMenus();
        List<Menu> list = listByIds(menuIds);
        List<Menu> result = new ArrayList<>();
        // 遍历菜单列表，根据角色ID和菜单ID查找对应的菜单
//        for (Menu menu : menus) {
//            if (menuIds.contains(menu.getId())) {
//                result.add(menu);
//            }
//            List<Menu> children = menu.getChildren();
//            children.removeIf(child -> !menuIds.contains(child.getId()));
//        }
        for (Menu menu : list) {
            if (menu.getParentId().equals(0L)) {
                result.add(menu);
            }
        }
        result = result.stream().sorted(Comparator.comparing(Menu::getSort)).collect(Collectors.toList());

        for (Menu menu : result) {
            List<Menu> children = new ArrayList<>();
            for (Menu item : list) {
                if (item.getParentId().equals(menu.getId())) {
                    children.add(item);
                }
            }
            List<Menu> sortList = children.stream().sorted(Comparator.comparing(Menu::getSort)).collect(Collectors.toList());
            menu.setChildren(sortList);
        }
        // 如果没有找到角色或菜单，返回空列表
        return result;
    }
}

