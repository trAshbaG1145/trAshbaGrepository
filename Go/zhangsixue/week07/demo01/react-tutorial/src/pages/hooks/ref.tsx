import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  memo,
  useMemo,
} from "react";


function Example5() {
  const inputEl = useRef<HTMLInputElement>(null);
  const [num, setNum] = useState<number>(1);
  const onButtonClick = () => {
    // `current` 指向已挂载到 DOM 上的文本输入元素
    inputEl.current?.focus();
  };

  const getInputValue = () => {
    console.log(inputEl.current?.value);
  };
  console.log("inputEl:", inputEl);
  return (
    <>
      <input ref={inputEl} type="text" />
      <button onClick={onButtonClick}>Focus the input</button>
      <button onClick={getInputValue}>Get input value</button>
      <button
        onClick={() => {
          setNum(num + 1);
        }}
      >
        {num}
      </button>
    </>
  );
}
function Example6() {
  const count = useRef<number>(0);
  const [, setForceUpdate] = useState({});

  const handleClick = () => {
    count.current += 1;
    console.log('Current count:', count.current);
    setForceUpdate({});
  };

  return (
    <div>
      <p>Count: {count.current}</p>
      <button onClick={handleClick}>Increment</button>
    </div>
  );
}

export default () => {
  const style:React.CSSProperties = {
    width: "500px",
    background: "#ccc",
    padding: "10px",
  };

  return (
    <div style={style}>
      <Example5 />
    <hr />
      <Example6 />
    </div>
  );
};
