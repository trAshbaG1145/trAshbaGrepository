import React, { useEffect, useState, useImperativeHandle, useRef } from "react";

import { Button, Modal, Form, Input } from "antd";

interface Iprops {
  open: (options?: UserInfo) => void;
  close: () => void;
}
interface UserInfo {
  name?: string;
  company?: string;
}

type FieldType = {
  name?: string;
  company?: string;
};

const ModalD = React.forwardRef<Iprops, { onOk?: (e: UserInfo) => void }>(
  (props, ref) => {
    const [isOpen, setIsOpen] = useState(false);
    const [userInfo, setUserInfo] = useState<UserInfo>({} as UserInfo);
    const [form] = Form.useForm();

    const title = Object.keys(userInfo).length > 0 ? "编辑数据" : "添加数据";

    useImperativeHandle(ref, () => ({
      open: (options?: UserInfo) => {
        setUserInfo(options || ({} as UserInfo));
        setIsOpen(true);
      },
      close: () => {
        setUserInfo({} as UserInfo);
        setIsOpen(false);
      },
    }));

    useEffect(() => {
      console.log("111");
      //initialValues  表单默认值，只有初始化以及重置时生效
      form.resetFields();
    }, [isOpen]);

    const handleCancel = () => {
      setIsOpen(false);
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
      props?.onOk?.(val);
      console.log("submitForm data:", val);
      form.resetFields();
      if (!val) return;
      setIsOpen(false);
    };

    return (
      <>
        {isOpen && (
          <Modal
            title={title}
            open={isOpen}
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
  }
);

export default function Main() {
  const modelRef = useRef<Iprops>(null);
  const handleAdd = () => {
    modelRef.current?.open();
  };
  const handleEdit = () => {
    const options: UserInfo = {
      name: "zhangsixue",
      company: "kingsoft",
    };
    modelRef.current?.open(options);
  };

  const onOk = (e: UserInfo) => {
    console.log("onOk", e);
  };
  console.log("Main");

  return (
    <div>
      <h1>弹窗组件 实现方式三</h1>
      <Button onClick={handleAdd}>添加</Button>
      <Button onClick={handleEdit}>编辑</Button>
      <ModalD ref={modelRef} onOk={onOk} />
    </div>
  );
}
