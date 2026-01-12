import React, { useState } from "react";

class Child extends React.Component {
  constructor(props) {
    super(props);
    this.state = { score: 1 };
    console.log('constructor')
  }

  componentWillMount() {
    console.log('componentWillMount')
  }

  componentDidMount() {
    console.log('componentDidMount')

  }
  componentWillReceiveProps(nextProps) {
    console.log('componentWillReceiveProps', nextProps)

  }

  shouldComponentUpdate(nextProps, nextState) {
    console.log("shouldComponentUpdate nextProps,nextState", nextProps, nextState)
    return true
  }

  componentWillUpdate() {
    console.log("componentWillUpdate")
  }

  componentDidUpdate(prevProps, prevState,) {
    console.log("componentDidUpdate prevProps, prevState:", prevProps, prevState,)

  }

  componentWillUnmount() {
    console.log("componentWillUnmount")

  }
  addScore = () => {
    this.setState({ score: this.state.score + 1 })
  }

  static getDerivedStateFromProps(nextProps, state) {
    console.log('nextProps, state:', nextProps, state)
  }

  getSnapshotBeforeUpdate(prevProps, prevState) {
    console.log('getSnapshotBeforeUpdate prevProps, prevState:', prevProps, prevState);
    return null;
  }


  render() {
    console.log('render')
    return (
      <div style={{ background: "red", padding: "5px" }}>
        <button>
          父组件属性num:{this.props.num}
        </button>
        <button>
          子组件属性score:{this.state.score}
        </button>
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
  const [num, setNum] = useState(1)

  const [closed, setClosed] = useState(false)
  return (
    <div style={style}>
      {
        !closed && <Child num={num} />
      }
      <hr />
      <button onClick={() => {
        setNum(num + 1)
      }}>Add num</button>
      <button onClick={() => {
        setClosed(true)
      }}>Set it closed</button>
    </div>
  );
};
