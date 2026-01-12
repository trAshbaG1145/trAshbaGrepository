import React, { useState } from "react";



function Welcome(props) {
  return <div>Welcome: Hello, {props.name}</div>;
}

class Welcome2 extends React.Component {
  constructor(props) {
    super(props)
  }
  render() {
    return <div>Welcome2: Hello, {this.props.name}</div>;
  }
}
class Example1 extends React.Component {
  render() {
    return (
      <>
        <Welcome name="Sara" />
        <Welcome2 name="Cahal" />
        <Welcome name="Edite" />
      </>
    );
  }
}

class Example2 extends React.Component {
  constructor(props) {
    super(props);
    this.state = { date: new Date() };
  }

  componentDidMount() {
    console.log('componentDidMount ')
    this.timerID = setInterval(
      () => this.tick(),
      1000
    );
  }

  componentWillUnmount() {
    clearInterval(this.timerID);
  }

  tick() {
    this.setState({
      date: new Date()
    });
  }

  render() {
    return (
      <div>
        <div>Hello, world!</div>
        <div>It is {this.state.date.toLocaleTimeString()}.</div>
      </div>
    );
  }
}

class Example3 extends React.Component {
  constructor(props) {
    super(props);
    this.myRef = React.createRef();
    console.log('this.myRef', this.myRef)
  }
  inputRef = React.createRef();
  render() {
    console.log('this.inputRef', this.inputRef)
    return (
      <>
        <div ref={this.myRef}>hello,world!</div>
        <input ref={this.inputRef} type="text" />
      </>
    )
  }
}

class Div1 extends React.Component {
  render() {
    console.log('Div1', this.props.num)
    return <p>{this.props.num}</p>;
  }
}
class Div2 extends React.PureComponent {
  render() {
    console.log('Div2', this.props.num)
    return <p>{this.props.num}</p>;
  }
}

class Div3 extends React.Component {
  shouldComponentUpdate(nextProps, nextState) {
    console.log('nextProps', nextProps)
    console.log('nextState', nextState)
    if (nextProps.num === this.props.num) {
      return false;
    }
    return true;
  }
  render() {
    console.log('Div3', this.props.num)
    return <p>{this.props.num}</p>;
  }
}

class Example4 extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      num: 0
    }
  }
  click = () => {
    this.setState({
      num: this.state.num + 1
    })
    // this.setState({
    //   num: 1
    // })
  }
  render() {
    return (
      <>
        <Div1 num={this.state.num}></Div1>
        <Div2 num={this.state.num}></Div2>
        <Div3 num={this.state.num}></Div3>

        <button onClick={this.click}>Click</button>
      </>
    )
  }
}


class App extends React.Component {
  render() {
    return (
      <>
        <Welcome name="Sara" />
        <hr />
        <Welcome2 name="Cahal" />
        <hr />
        <Example1 />
        <hr />
        <Example2 />
        <hr />
        <Example3 />
        <hr />
        <Example4 />
      </>
    );
  }
}

export default App