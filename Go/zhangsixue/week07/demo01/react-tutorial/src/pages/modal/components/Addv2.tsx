import React, {useState } from "react";
import { Button, Modal, Form, Input } from "antd";

export interface UserInfo {
  name: string;
  company: string;
}

interface IProps {
  children: React.ReactNode;
  userInfo?: UserInfo;
}

type FieldType = {
  name?: string;
  company?: string;
};

const Add: React.FC<IProps> = (props) => {
  const { children, userInfo = {} } = props;
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

  const title = Object.keys(userInfo).length > 0 ? "编辑数据" : "添加数据";

  const [form] = Form.useForm();

  console.log("userInfo:", userInfo);
  const showModal = () => {
    setIsModalOpen(true);
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
    setIsModalOpen(false);
  };

  const handleCancel = () => {
    setIsModalOpen(false);
  };

  return (
    <>
      <Button onClick={showModal}>{children}</Button>
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
