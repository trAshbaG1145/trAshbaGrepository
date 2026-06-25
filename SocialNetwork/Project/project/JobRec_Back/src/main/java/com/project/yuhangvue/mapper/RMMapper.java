package com.project.yuhangvue.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.project.yuhangvue.entity.RoleMenu;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface RMMapper extends BaseMapper<RoleMenu> {

    @Select("select menu_id from role_menu where role_id = #{roleId}")
    List<Long> findMenuIdsByRoleId(Long roleId);
}
