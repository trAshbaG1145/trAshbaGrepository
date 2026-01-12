import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  memo,
  useMemo,
} from "react";

function Example1() {
  // 声明一个叫 "count" 的 state 变量。setCount是异步的
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>Click me</button>
    </div>
  );
}

interface Example2Props {}

interface Example2State {
  count: number;
}

class Example2 extends React.Component<Example2Props, Example2State> {
  constructor(props: Example2Props) {
    super(props);
    this.state = {
      count: 0,
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
  // 声明一个叫 "count" 的 state 变量
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => {
        // console.log('setCount', count)
        // setCount(count + 1)
        // console.log('setCount', count)
        // setCount(count + 1)
        // console.log('setCount', count)
        // setCount(count + 1)
        setCount(count => count + 1)
        setCount(count => count + 1)
        setCount(count => count + 1)

      }}>Click me 3</button>
    </div>
  );
}

export default () => {
  const style = {
    width: "500px",
    background: "#ccc",
    padding: "10px",
  };

  return (
    <div style={style}>
      1:
      <Example1 />
      <hr />
      2:
      <Example2 />
      <hr />
      3:
      <Example3 />
    </div>
  );
};
