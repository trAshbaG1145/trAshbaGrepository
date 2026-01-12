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

function Example7() {
  const [num, setNum] = useState(1);

  const addOne = () => {
    setNum((num) => num + 1);
  };
  return (
    <>
      <div>{num}</div>
      <Button2 onClick={addOne}>Add</Button2>
    </>
  );
}

function Example8() {
  const [num, setNum] = useState(1);

  const addOne = () => {
    setNum((num) => num + 1);
  };
  const button3 = useMemo(
    () => (
      <Button2 onClick={addOne}>Add</Button2>
    ),
    []
  )
  return (
    <>
      <div>{num}</div>
      {button3}
    </>
  );
}

export default () => {
  return (
    <div>
      7:
      <Example7 />
      <hr />
      8:
      <Example8 />
    </div>
  );
};
