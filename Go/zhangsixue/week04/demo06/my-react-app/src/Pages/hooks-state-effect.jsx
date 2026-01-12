import React, { useState, useEffect, useRef, useCallback, memo, useMemo } from "react";

function Example1() {
  // 声明一个叫 "count" 的 state 变量
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}

class Example2 extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      count: 0
    };
  }

  render() {
    return (
      <div>
        <p>You clicked {this.state.count} times</p>
        <button onClick={() => this.setState({ count: this.state.count + 1 })}>
          Click me
        </button>
      </div>
    );
  }
}


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
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}

function Example4() {
  const [count, setCount] = useState(0);

  // 与 componentDidMount() 和 componentDidUpdate() 类似:
  useEffect(() => {
    // 使用浏览器 API 更新网页标题
    document.title = `You clicked ${count} times`;
    return () => {
      console.log('close me')
    }
  });

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}

function Example5() {
  const inputEl = useRef(null);
  const [num, setNum] = useState(1)
  const onButtonClick = () => {
    // `current` 指向已挂载到 DOM 上的文本输入元素
    inputEl.current.focus();
  };

  const getInputValue = () => {
    console.log(inputEl.current.value)
  }
  console.log('inputEl:', inputEl)
  return (
    <>
      <input ref={inputEl} type="text" />
      <button onClick={onButtonClick}>Focus the input</button>
      <button onClick={getInputValue}>Get input value</button>
      <button onClick={() => {
        setNum(num + 1)
      }}>{num}</button>
    </>
  );
}


export default () => {
  const style = {
    width: "500px",
    background: "#ccc",
    padding: "10px",
  };

  const [close, setClose] = useState(false)
  return (
    <div style={style}>
      1:<Example1 />
      <hr />
      2:<Example2 />
      <hr />
      3:<Example3 />
      <hr />
      4:
      {
        !close && <Example4 />
      }
      <button onClick={() => {
        setClose(true)
      }}>close Example4</button>
      <hr />
      5:<Example5 />

    </div>
  );
};
