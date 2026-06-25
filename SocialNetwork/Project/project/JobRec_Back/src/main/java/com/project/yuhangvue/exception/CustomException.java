package com.project.yuhangvue.exception;/*
 *   @Author:田宇航
 *   @Date: 2025/1/4 20:13
 */

import lombok.Data;
import lombok.Getter;
@Data
@Getter
public class CustomException extends RuntimeException {
    private Integer code;
    private String msg;

    public CustomException(Integer code, String msg) {
        super(msg);
        this.code = code;
    }

    public CustomException(String msg) {
        super(msg);
        this.code = 500;
    }
    public static class Errors {
        public static final CustomException USER_NOT_FOUND = new CustomException(301, "用户名或密码错误");
        public static final CustomException USER_DISABLED = new CustomException(302, "访问受限");
        public static final CustomException UPDATE_ERROR = new CustomException(303, "修改失败");
        public static final CustomException ADD_ERROR = new CustomException(304, "添加失败");
        public static final CustomException DELETE_ERROR = new CustomException(305, "删除失败");
        public static final CustomException PARAM_ERROR = new CustomException(306, "参数错误");
        public static final CustomException SAVE_ERROR = new CustomException(307, "保存失败");
        public static final CustomException INVALID_CREDENTIALS = new CustomException(310, "密码错误");
        public static final CustomException USER_DELETED = new CustomException(311, "用户不存在");



    }


}

