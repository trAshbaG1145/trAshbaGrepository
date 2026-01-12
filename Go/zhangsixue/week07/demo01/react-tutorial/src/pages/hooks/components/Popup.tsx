import React, {
  useImperativeHandle,
  forwardRef,
  useState,
  useCallback,
} from "react";
import { Modal } from "antd";

export interface PopupRef {
  openPopup: () => void;
  closePopup: () => void;
}
export interface PopupProps {
  onOk?: (s:string) => void;
}

const Popup = forwardRef<PopupRef, PopupProps>((props, ref) => {
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

  useImperativeHandle(ref, () => ({
    openPopup: () => {
      console.log("弹窗打开");
      setIsModalOpen(true);
      // 弹窗打开的逻辑
    },
    closePopup: () => {
      console.log("弹窗关闭");
      setIsModalOpen(false);

      // 弹窗关闭的逻辑
    },
  }));
  const onOk = useCallback(() => {
    props.onOk?.("something");
    setIsModalOpen(false);
  }, []);

  const onCancel = useCallback(() => {
    setIsModalOpen(false);
  }, []);

  return (
    <Modal open={isModalOpen} onOk={onOk} onCancel={onCancel}>
      <div>这里是弹窗的内容</div>
    </Modal>
  );
});

export default Popup;
