import React from "react";
import Add, { UserInfo } from "./components/Addv2";

import { Space } from "antd";

export default function Main() {
  const addInfo: UserInfo = {} as UserInfo;
  const editInfo: UserInfo = { name: "zhangsan2", company: "alibaba2" };
  return (
    <div>
      <h1>弹窗组件 实现方式二</h1>
      <Space>
        <Add userInfo={addInfo}>添加</Add>
        <Add userInfo={editInfo}>编辑</Add>
      </Space>
    </div>
  );
}
