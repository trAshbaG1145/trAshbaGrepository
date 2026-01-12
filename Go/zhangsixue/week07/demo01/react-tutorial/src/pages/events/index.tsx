import React from "react";

//错误案例
class Example1 extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isToggleOn: true };
  }

  handleClick() {
    console.log("Toggle1 this", this);
    this.setState((prevState) => ({
      isToggleOn: !prevState.isToggleOn,
    }));
  }

  render() {
    return (
      <button onClick={this.handleClick}>
        {this.state.isToggleOn ? "ON" : "OFF"}
      </button>
    );
  }
}
class Toggle2 extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isToggleOn: true };

    // 为了在回调中使用 `this`，这个绑定是必不可少的
    this.handleClick = this.handleClick.bind(this);
  }

  handleClick() {
    console.log("Toggle2 this", this);
    this.setState((prevState) => ({
      isToggleOn: !prevState.isToggleOn,
    }));
  }

  render() {
    return (
      <button onClick={this.handleClick}>
        {this.state.isToggleOn ? "ON" : "OFF"}
      </button>
    );
  }
}
class Toggle3 extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isToggleOn: true };
  }

  handleClick = () => {
    console.log("Toggle3 this", this);
    this.setState((prevState) => ({
      isToggleOn: !prevState.isToggleOn,
    }));
  };

  render() {
    return (
      <button onClick={this.handleClick}>
        {this.state.isToggleOn ? "ON" : "OFF"}
      </button>
    );
  }
}

class Toggle4 extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isToggleOn: true };
  }

  render() {
    return (
      <button
        onClick={() => {
          console.log("Toggle4 this", this);
          this.setState((prevState) => ({
            isToggleOn: !prevState.isToggleOn,
          }));
        }}
      >
        {this.state.isToggleOn ? "ON" : "OFF"}
      </button>
    );
  }
}

class List extends React.Component {
  arr = [
    {
      id: "001",
      name: "aaa",
    },
    {
      id: "002",
      name: "bbb",
    },
    {
      id: "003",
      name: "ccc",
    },
  ];

  deleteRow = (id) => {
    console.log("id:", id);
  };
  render() {
    return (
      <ul>
        {this.arr.map(({ id, name }) => {
          return (
            <li key={id} onClick={(e) => this.deleteRow(id, e)}>
              {name}
            </li>
          );
        })}
      </ul>
    );
  }
}

export default () => {
  const style = {
    width: "500px",
    background: "#ccc",
    padding: "10px",
  };
  return (
    <div style={style}>
      <Example1 />
      <hr />
      <Toggle2 />
      <hr />
      <Toggle3 />
      <hr />
      <Toggle4 />
      <hr />
      <List />
    </div>
  );
};
