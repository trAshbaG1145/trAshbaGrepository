import React, { useRef } from 'react';
import Popup,{PopupRef} from './components/Popup';

function App() {
  const popupRef = useRef<PopupRef>({} as PopupRef);

  const openPopup = () => {
    popupRef.current?.openPopup();
  };

  const closePopup = () => {
    popupRef.current?.closePopup();
  };

  return (
    <div>
      <h1>hook组件中使用 ref </h1>
      <button onClick={openPopup}>打开弹窗</button>
      <button onClick={closePopup}>关闭弹窗</button>
      <Popup onOk={(str)=>{
        console.log("onOk",str);
      }} ref={popupRef} />
    </div>
  );
}

export default App;
