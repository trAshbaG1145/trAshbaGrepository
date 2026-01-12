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

//正常情况，父组件渲染， 子组件都会重新渲染 Button2 会一直渲染
function Button2(props: Props) {
  console.log("Button2 render");
  return <button onClick={props.onClick}>{props.children}</button>;
}

function Example1() {
  const [num, setNum] = useState(1);

  const addOne = () => {
    setNum((num) => num + 1);
  };
  return (
    <>
      <div>{num}</div>
      <button onClick={addOne}>Add</button>
      <Button2>Button2</Button2>
    </>
  );
}

// 优化方式：const Button3 = memo(Button2);，但是要求Button3 组件的 props 不变
const Button3 = memo(Button2);

function Example2() {
  const [num, setNum] = useState(1);

  const addOne = () => {
    setNum((num) => num + 1);
  };
  return (
    <>
      <div>{num}</div>
      <button onClick={addOne}>Add</button>
      <Button3>Button3</Button3>
    </>
  );
}

export default () => {
  return (
    <div>
      1：<Example1 />
      <hr />
      2：<Example2 />
    </div>
  );
};
