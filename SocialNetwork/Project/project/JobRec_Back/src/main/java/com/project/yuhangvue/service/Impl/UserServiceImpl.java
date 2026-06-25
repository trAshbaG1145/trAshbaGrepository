package com.project.yuhangvue.service.Impl;/*
 *   @Author:田宇航
 *   @Date: 2025/4/21 14:27
 */

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.StringUtils;
import com.project.yuhangvue.common.Result;
import com.project.yuhangvue.dto.LoginDTO;
import com.project.yuhangvue.entity.*;
import com.project.yuhangvue.exception.CustomException;
import com.project.yuhangvue.mapper.*;
import com.project.yuhangvue.service.MenuService;
import com.project.yuhangvue.service.UserService;
import com.project.yuhangvue.utils.Argon2PasswordEncoder;
import com.project.yuhangvue.utils.JwtUtil;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.List;
import java.util.UUID;

import static com.project.yuhangvue.common.Result.success;
@Slf4j
@Service
public class UserServiceImpl implements UserService {
    @Resource
    private CandidateMapper candidateMapper;

    @Resource
    private MenuService menuService;

    @Resource
    private CandidateCardMapper candidateCardMapper;

    @Resource
    private FirmMapper firmMapper;

    @Resource
    private FirmCardMapper firmCardMapper;
    @Resource
    private URMapper urMapper;
    @Override
    public Result login(@RequestBody LoginDTO loginDTO) {
        String username = loginDTO.getUsername();
        String password = loginDTO.getPassword();

        // 第一步：根据用户名查找求职者
        LambdaQueryWrapper<Candidate> candidateWrapper = new LambdaQueryWrapper<>();
        candidateWrapper.eq(Candidate::getUsername, username).last("limit 1");
        Candidate candidate = candidateMapper.selectOne(candidateWrapper);

        // 如果是求职者，执行验证逻辑
        if (candidate != null) {
            return validateUserLogin(candidate, password, true);
        }

        // 第二步：根据用户名查找公司
        LambdaQueryWrapper<Firm> firmWrapper = new LambdaQueryWrapper<>();
        firmWrapper.eq(Firm::getUsername, username).last("limit 1");
        Firm firm = firmMapper.selectOne(firmWrapper);

        // 如果是公司，执行验证逻辑
        if (firm != null) {
            return validateUserLogin(firm, password, false);
        }

        // 如果用户名既不是求职者也不是公司，抛出用户不存在异常
        throw CustomException.Errors.USER_NOT_FOUND;
    }

    @Override
    public Result getInfo(Long userId, Integer scope) {
        try {
            if (scope == 1) {
                Candidate candidate = candidateMapper.selectById(userId);
                return Result.success(candidate);
            }
            if (scope == 2) {
                Firm firm = firmMapper.selectById(userId);
                return Result.success(firm);
            } else {
                return Result.error("用户类型错误");
            }
        } catch (Exception e) {
            return Result.error(e.getMessage());
        }
    }

    @Override
    public Result<String> updatePassword(Long userId, String oldPassword, String newPassword, Integer scope) {
        try {
            if (scope == 1) {
                Candidate candidate = candidateMapper.selectById(userId);
                if (!Argon2PasswordEncoder.matches(candidate.getPassword(), oldPassword)) {
                    return Result.error("原密码错误");
                }
                candidate.setPassword(Argon2PasswordEncoder.encode(newPassword));
                candidateMapper.updateById(candidate);
                return Result.success();
            }
            if (scope == 2) {
                Firm firm = firmMapper.selectById(userId);
                if (!Argon2PasswordEncoder.matches(firm.getPassword(), oldPassword)) {
                    return Result.error("原密码错误");
                }
                firm.setPassword(Argon2PasswordEncoder.encode(newPassword));
                firmMapper.updateById(firm);
                return Result.success();
            } else {
                return Result.error("用户类型错误");
            }
        } catch (Exception e) {
            return Result.error(e.getMessage());
        }
    }

    @Override
    public Result register(LoginDTO registerDTO) {
        try {
            String username = registerDTO.getUsername();
            String password = registerDTO.getPassword();
            String email = registerDTO.getEmail();
            Integer scope = registerDTO.getScope();
            if (scope == 1) {
                Candidate candidate = new Candidate();
                candidate.setId(UUID.randomUUID().getMostSignificantBits());
                candidate.setUsername(username);
                candidate.setPassword(Argon2PasswordEncoder.encode(password));
                candidate.setEmail(email);
                candidateMapper.insert(candidate);
                UserRole userRole = new UserRole();
                userRole.setUserId(candidate.getId());
                userRole.setRoleId(1L);
                urMapper.insert(userRole);
                String token = JwtUtil.createToken(candidate);
                return Result.success(new LoginDTO(token, candidate.getId(), getMenu(candidate.getId()), false, scope));
            } else if (scope == 2) {
                Firm firm = new Firm();
                firm.setId(UUID.randomUUID().getMostSignificantBits());
                firm.setUsername(username);
                firm.setPassword(Argon2PasswordEncoder.encode(password));
                firm.setEmail(email);
                firmMapper.insert(firm);
                UserRole userRole = new UserRole();
                userRole.setUserId(firm.getId());
                userRole.setRoleId(2L);
                urMapper.insert(userRole);
                String token = JwtUtil.createToken(firm);
                return Result.success(new LoginDTO(token, firm.getId(), getMenu(firm.getId()), false, scope));

            } else {
                return Result.error("用户类型错误");
            }
        } catch (Exception e) {
            return Result.error(e.getMessage());
        }
    }

    private Result validateUserLogin(User user, String password, boolean isCandidate) {
        // 检查密码是否匹配
        if (!Argon2PasswordEncoder.matches(user.getPassword(), password)) {
            throw CustomException.Errors.INVALID_CREDENTIALS;
        }

        // 检查用户状态是否启用
        if (user.getStatus() != 1) {
            throw CustomException.Errors.USER_DISABLED;
        }

        // 检查用户是否未删除
        if (user.getIsDeleted() != 0) {
            throw CustomException.Errors.USER_DELETED;
        }

        // 生成JWT令牌
        String token;
        Integer scope;
        boolean hasCard;
        if (isCandidate) {
            hasCard = candidateCardMapper.selectCount(new LambdaQueryWrapper<CandidateCard>().eq(CandidateCard::getCandidateId, user.getId())) > 0;
            scope = 1;
            token = JwtUtil.createToken((Candidate) user);
            // 返回登录结果
        } else {
            hasCard = firmCardMapper.selectCount(new LambdaQueryWrapper<FirmCard>().eq(FirmCard::getFirmId, user.getId())) > 0;
            scope = 2;
            token = JwtUtil.createToken((Firm) user);
            // 返回登录结果
        }
        return success(new LoginDTO(token, user.getId(), getMenu(user.getId()), hasCard, scope));
    }

    // 获取菜单方法
    private List<Menu> getMenu(Long userId) {
        return menuService.getMenu(userId);
    }

    @Override
    public Result updateAvatar(Long id, Integer scope, String avatar) {
        try {
            // 前置参数校验
            if (id == null || scope == null || StringUtils.isBlank(avatar)) {
                return Result.error("缺少必要参数");
            }

            if (scope == 1) {
                Candidate candidate = candidateMapper.selectById(id);
                if (candidate == null) {
                    return Result.error("候选人不存在");
                }
                candidate.setAvatar(avatar);
                candidateMapper.updateById(candidate);
            } else if (scope == 2) {
                Firm firm = firmMapper.selectById(id);
                if (firm == null) {
                    return Result.error("企业不存在");
                }
                firm.setAvatar(avatar);
                firmMapper.updateById(firm);
            } else {
                return Result.error("无效的用户类型");
            }

            return Result.success();
        } catch (Exception e) {
            log.error("头像更新失败: {}", e.getMessage(), e);  // 添加日志记录
            return Result.error("系统繁忙，请稍后再试");  // 避免暴露具体异常信息
        }
    }
}

