import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  memo,
  useMemo,
} from "react";

function Example3() {
  const [count, setCount] = useState(0);

  // 与 componentDidMount() 和 componentDidUpdate() 类似:
  useEffect(() => {
    // 使用浏览器 API 更新网页标题
    document.title = `You clicked ${count} times`;
  });

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>Click me</button>
    </div>
  );
}

function Example4() {
  const [count, setCount] = useState(0);

  // 与 componentDidMount() 和 componentDidUpdate() 类似:
  useEffect(() => {
    // 使用浏览器 API 更新网页标题
    document.title = `You clicked ${count} times`;
  }, [count]);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button
        onClick={() => {
          if (Math.random() < 1 / 2) {
            setCount(count + 1);
          }
        }}
      >
        Click me
      </button>
    </div>
  );
}

function Example5() {
  const [count, setCount] = useState(0);

  // 与 componentDidMount() 和 componentDidUpdate() 类似:
  useEffect(() => {
    // 使用浏览器 API 更新网页标题
    document.title = `You clicked ${count} times`;
    return () => {
      console.log("close me");
    };
  });

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>Click me</button>
    </div>
  );
}

export default () => {
  const style = {
    width: "500px",
    background: "#ccc",
    padding: "10px",
  };
  const [close, setClose] = useState(false);

  return (
    <div style={style}>
      3:
      <Example3 />
      <hr />
      4:
      <Example4 />
      <hr />
      5:
      {!close && <Example5 />}
      <button
        onClick={() => {
          setClose(true);
        }}
      >
        close Example5
      </button>
      <hr />
    </div>
  );
};
