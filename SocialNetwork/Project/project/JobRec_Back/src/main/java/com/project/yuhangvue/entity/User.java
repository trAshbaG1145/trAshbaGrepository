package com.project.yuhangvue.entity;


import java.util.Date;

public interface User {
    Long getId();
    String getUsername();
    String getPassword();
    String getNickname();
    String getAvatar();
    String getPhone();
    String getEmail();
    Date getCreateTime();
    Date getUpdateTime();
    Integer getIsDeleted();
    Integer getStatus();
    Integer getGender();
    String getName();


}
