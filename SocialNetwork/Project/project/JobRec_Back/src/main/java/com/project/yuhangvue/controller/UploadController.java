package com.project.yuhangvue.controller;/*
 *   @Author:田宇航
 *   @Date: 2025/4/12 07:47
 */

import com.project.yuhangvue.common.Result;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

@RestController
@RequestMapping("/file")
public class UploadController {
    @PostMapping(path = "/upload", consumes = "multipart/form-data")
    public Result<String> handleFileUpload(@RequestParam("file") MultipartFile file) {
        try {
            // 文件名随机生成
            String originalFilename = file.getOriginalFilename();
            // 检查 originalFilename 是否为 null，若为 null 则使用空字符串作为扩展名
            String fileExtension = originalFilename != null ? originalFilename.substring(originalFilename.lastIndexOf(".")) : "";
            String randomFilename = UUID.randomUUID() + fileExtension;
            Path uploadDir = Path.of("uploaded-files", "resume");
            Files.createDirectories(uploadDir);
            file.transferTo(uploadDir.resolve(randomFilename).toFile());
            return Result.success(randomFilename);
        } catch (Exception e) {
            return Result.error("文件上传失败" );
        }
    }
    @PostMapping(path = "/uploadAvatar", consumes = "multipart/form-data")
    public Result<String> handleFileUploadAvatar(@RequestParam("file") MultipartFile file) {
        try {
            // 文件名随机生成
            String originalFilename = file.getOriginalFilename();
            String fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
            String randomFilename = UUID.randomUUID() + fileExtension;
            Path uploadDir = Path.of("uploaded-files", "images", "uploads");
            Files.createDirectories(uploadDir);
            file.transferTo(uploadDir.resolve(randomFilename).toFile());
            return Result.success(randomFilename);
        }
        catch (Exception e) {
            return Result.error("文件上传失败" );
        }
    }
}

