import React, { useRef, useState } from "react";

interface IProps {
  name: string;
}

interface IState {
  count: number;
}

class Welcome1 extends React.Component<IProps, IState> {
  state = {
    count: 0,
  };

  add = () => {
    this.setState({
      count: this.state.count + 1,
    });
  };
  render() {
    return (
      <div>
        {this.state.count}
        {this.props.name}
        <button onClick={this.add}>Add</button>
      </div>
    );
  }
}

const Welcome2 = (props: IProps) => {
  const { name } = props;

  return (
    <div className="App">
      <div>hello world</div>
      <div>{name}</div>
    </div>
  );
};

const Welcome3: React.FC<IProps> = (props) => {
  const { name } = props;
  const [count, setCount] = useState<number>(0);
  const inputref = useRef<HTMLInputElement>(null);

  const add = () => {
    setCount(count + 1);
  };
  const getValue = () => {
    console.log(inputref.current?.value);
  };
  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log(e.target.value);
  };
  return (
    <div className="App">
      <div>{name}</div>
      <div>{count}</div>
      <button onClick={add}>add</button>
      <input type="text" ref={inputref} onChange={onChange} />
      <button onClick={getValue}>getValue</button>
    </div>
  );
};

function Welcome4(props: IProps) {
  return <div>{props.name}</div>;
}

class App extends React.Component {
  render() {
    return (
      <>
        <h1> 组件中使用TS</h1>
        <Welcome1 name="Sara" />
        <hr />
        <Welcome2 name="Cahal" />
        <hr />
        <Welcome3 name="Lucy" />
        <hr />
        <Welcome4 name="Kate" />
      </>
    );
  }
}

export default App;
