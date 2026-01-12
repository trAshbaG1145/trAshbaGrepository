import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  memo,
  useMemo,
} from "react";

export interface Props {
  onClick?: () => void;
  children?: React.ReactNode;
}

//正常情况，父组件渲染， 子组件都会重新渲染
function Button2(props: Props) {
  console.log("Button2 render");
  return <button onClick={props.onClick}>{props.children}</button>;
}
const Button3 = memo(Button2);

function Example3() {
  const [num, setNum] = useState(1);

  // 每次渲染addOne 都是新的，Button3的props变化
  const addOne = () => {
    setNum((num) => num + 1);
  };
  (window as any).addOne = addOne;
  return (
    <>
      <div>{num}</div>
      <Button3 onClick={addOne}>Add</Button3>
    </>
  );
}

function Example4() {
  const [num, setNum] = useState(1);

  // 新手经常踩得坑
  const addOne = useCallback(() => {
    setNum(num + 1);
  }, []);
  (window as any).addOne = addOne;
  return (
    <>
      <div>{num}</div>
      <Button2 onClick={addOne}>Add</Button2>
    </>
  );
}

function Example5() {
  const [num, setNum] = useState(1);
  // 需要把依赖加到这，但是会导致addOne每次变化
  const addOne = useCallback(() => {
    setNum(num + 1);
  }, [num]);
  (window as any).addOne = addOne;
  return (
    <>
      <div>{num}</div>
      <Button2 onClick={addOne}>Add</Button2>
    </>
  );
}


function Example6() {
  const [num, setNum] = useState(1);
  // 利用state的函数式更新
  const addOne = useCallback(() => {
    setNum(num => num + 1);
  }, []);
  return (
    <>
      <div>{num}</div>
      <Button2 onClick={addOne}>Add</Button2>
    </>
  );
}

export default () => {
  return (
    <div>
      3：<Example3 />
      <hr />
      4：<Example4 />
      <hr />
      5：<Example5 />
      <hr />
      6：<Example6 />
    </div>
  );
};
