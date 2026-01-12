import React, { ComponentType, useState, ReactNode, useEffect } from "react";
import Add, { UserInfo } from "./components/Addv1";
import { Button } from "antd";

export default function Main() {
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [userInfo, setUserInfo] = useState<UserInfo>({} as UserInfo);

  const showModal = () => {
    setUserInfo({} as UserInfo);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setUserInfo({} as UserInfo);
    setIsModalOpen(false);
  };

  const showEdit = () => {
    setUserInfo({ name: "zhangsan", company: "alibaba" });
    setIsModalOpen(true);
  };

  return (
    <div>
      <h1>弹窗组件 实现方式一</h1>
      <Button onClick={showModal}>添加</Button>
      <Add
        isModalOpen={isModalOpen}
        userInfo={userInfo}
        closeModal={closeModal}
      />
      <Button onClick={showEdit}>编辑</Button>
    </div>
  );
}
