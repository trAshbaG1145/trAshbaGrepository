import React, { useState } from "react";

// 定义 props 和 state 的接口
interface ChildProps {
  num: number;
}

interface ChildState {
  score: number;
}

class Child extends React.Component<ChildProps, ChildState> {
  constructor(props: ChildProps) {
    super(props);
    this.state = { score: 1 };
    console.log("constructor");
  }

  componentWillMount() {
    console.log("componentWillMount");
  }

  componentDidMount() {
    console.log("componentDidMount");
  }
  
  componentWillReceiveProps(nextProps: ChildProps) {
    console.log("componentWillReceiveProps", nextProps);
  }

  // 返回true 则永远重新渲染
  shouldComponentUpdate(nextProps: ChildProps, nextState: ChildState) {
    console.log(
      "shouldComponentUpdate nextProps,nextState",
      nextProps,
      nextState
    );
    return true;
  }

  componentWillUpdate() {
    console.log("componentWillUpdate");
  }

  componentDidUpdate(prevProps: ChildProps, prevState: ChildState) {
    console.log(
      "componentDidUpdate prevProps, prevState:",
      prevProps,
      prevState
    );
  }

  componentWillUnmount() {
    console.log("componentWillUnmount");
  }
  
  addScore = () => {
    this.setState({ score: this.state.score + 1 });
  };

  static getDerivedStateFromProps(nextProps: ChildProps, state: ChildState) {
    console.log("nextProps, state:", nextProps, state);
    return null;
  }

  getSnapshotBeforeUpdate(prevProps: ChildProps, prevState: ChildState) {
    console.log(
      "getSnapshotBeforeUpdate prevProps, prevState:",
      prevProps,
      prevState
    );
    return null;
  }

  render() {
    console.log("render");
    return (
      <div style={{ background: "red", padding: "5px" }}>
        <button>父组件属性num:{this.props.num}</button>
        <button>子组件属性score:{this.state.score}</button>
        <button onClick={this.addScore}>Add score</button>
      </div>
    );
  }
}

export default () => {
  const style = {
    width: "500px",
    background: "#ccc",
    padding: "10px",
  };
  const [num, setNum] = useState(1);

  const [closed, setClosed] = useState(false);
  return (
    <div style={style}>
      {!closed && <Child num={num} />}
      <hr />
      <button
        onClick={() => {
          setNum(num + 1);
        }}
      >
        Add num
      </button>
      <button
        onClick={() => {
          setClosed(true);
        }}
      >
        Set it closed
      </button>
    </div>
  );
};
