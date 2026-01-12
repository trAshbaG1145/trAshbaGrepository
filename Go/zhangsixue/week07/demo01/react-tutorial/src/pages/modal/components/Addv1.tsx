import React, { ComponentType, useState, ReactNode, useEffect } from "react";
import { Button, Modal, Form, Input } from "antd";

export interface UserInfo {
  name: string;
  company: string;
}

interface IProps {
  isModalOpen: boolean;
  closeModal: () => void;
  userInfo?: UserInfo;
}

type FieldType = {
  name?: string;
  company?: string;
};

const Add: React.FC<IProps> = (props) => {
  const { isModalOpen, closeModal, userInfo = {} } = props;
  console.log("userInfo:", userInfo);

  const [form] = Form.useForm();

  const title = Object.keys(userInfo).length > 0 ? "编辑数据" : "添加数据";

  useEffect(() => {
    console.log('111')
    //initialValues  表单默认值，只有初始化以及重置时生效	
    form.resetFields();
  } , [isModalOpen])

  const handleCancel = () => {
    closeModal();
  };

  const validate = () => {
    return form
      .validateFields()
      .then((values) => {
        return {
          ...values,
        };
      })
      .catch((err) => {
        console.log(err);
      });
  };

  const handleOk = async () => {
    const val = await validate();
    console.log("submitForm data:", val);
    form.resetFields();
    if (!val) return;
    closeModal();
  };


  return (
    <>
      {isModalOpen && (
        <Modal
          title={title}
          open={isModalOpen}
          onOk={handleOk}
          onCancel={handleCancel}
        >
          <Form
            form={form}
            name="basic"
            labelCol={{ span: 6 }}
            wrapperCol={{ span: 18 }}
            style={{ maxWidth: 600 }}
            initialValues={userInfo}
            autoComplete="off"
          >
            <Form.Item<FieldType>
              label="姓名"
              name="name"
              rules={[{ required: true, message: "Please input your name!" }]}
            >
              <Input />
            </Form.Item>

            <Form.Item<FieldType>
              label="公司名称"
              name="company"
              rules={[
                { required: true, message: "Please input your company!" },
              ]}
            >
              <Input />
            </Form.Item>
          </Form>
        </Modal>
      )}
    </>
  );
};

export default Add;
