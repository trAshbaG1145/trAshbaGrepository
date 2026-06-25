package com.project.yuhangvue.exception;/*
 *   @Author:田宇航
 *   @Date: 2025/1/4 20:11
 */

import com.project.yuhangvue.common.Result;
import lombok.extern.slf4j.Slf4j;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.List;

@Slf4j
@ControllerAdvice
public class GlobalExceptionHanlder {

    @ExceptionHandler(RuntimeException.class)
    @ResponseBody
    public Result<?> exceptionHandler(){
        return Result.error();
    }

    @ExceptionHandler(value = CustomException.class)
    @ResponseBody
    public Result<?> CustomExceptionHandler(CustomException e){
        log.error("错误原因:" + e.getMessage());
        return Result.error(e.getCode(),e.getMessage());
    }

    @ExceptionHandler(value = MethodArgumentNotValidException.class)
    @ResponseBody
    public Result<?> MethodArgumentNotValidExceptionHandler(MethodArgumentNotValidException e){
        List<FieldError> fieldErrors = e.getBindingResult().getFieldErrors();
        String defaultMessage = fieldErrors.get(0).getDefaultMessage();
        log.error("错误原因" + defaultMessage);
        return Result.error(defaultMessage);
    }

}

