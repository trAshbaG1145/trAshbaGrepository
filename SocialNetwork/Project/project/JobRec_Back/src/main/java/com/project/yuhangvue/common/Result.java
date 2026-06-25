package com.project.yuhangvue.common;



import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 统一结果封装类
 * @param <T> 数据类型
 */
@Data
@NoArgsConstructor
public class Result<T> {
    /**
     * 状态码
     */
    private Integer code;
    /**
     * 消息
     */
    private String message;
    /**
     * 数据
     */
    private T data;

    /**
     * 默认成功返回，无数据
     * @return 结果对象
     */
    public static <T> Result<T> success() {
        return new Result<>(200, "操作成功", null);
    }

    /**
     * 成功返回，带数据
     * @param data 数据
     * @param <T> 数据类型
     * @return 结果对象
     */
    public static <T> Result<T> success(T data) {
        return new Result<>(200, "操作成功", data);
    }

    /**
     * 自定义成功返回，带状态码、消息和数据
     * @param code 状态码
     * @param message 消息
     * @param data 数据
     * @param <T> 数据类型
     * @return 结果对象
     */
    public static <T> Result<T> success(int code, String message, T data) {
        return new Result<>(code, message, data);
    }

    /**
     * 默认失败返回，无数据
     * @return 结果对象
     */
    public static <T> Result<T> error() {
        return new Result<>(500, "操作失败", null);
    }

    /**
     * 失败返回，带消息，无数据
     * @param message 消息
     * @param <T> 数据类型
     * @return 结果对象
     */
    public static <T> Result<T> error(String message) {
        return new Result<>(500, message, null);
    }

    /**
     * 自定义失败返回，带状态码、消息和数据
     * @param code 状态码
     * @param message 消息
     * @param data 数据
     * @param <T> 数据类型
     * @return 结果对象
     */
    public static <T> Result<T> error(int code, String message, T data) {
        return new Result<>(code, message, data);
    }
    public Result(Integer code, String message, T data) {
        this.code = code;
        this.message = message;
        this.data = data;
    }

    public static Result<?> error(Integer code, String message) {
        return new Result<>(code, message, null);
    }
}