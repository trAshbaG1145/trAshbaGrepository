import React from "react";  
  

  
interface IDivProps {  
  num: number;  
}  
  
// props没有变化时，只要父组件重新渲染，则子组件重新渲染（hook也会这样）
class Div1 extends React.Component<IDivProps> {  
  render() {  
    console.log("Div1", this.props.num);  
    return <p>{this.props.num}</p>;  
  }  
}  
  
class Div2 extends React.PureComponent<IDivProps> {  
  render() {  
    console.log("Div2", this.props.num);  
    return <p>{this.props.num}</p>;  
  }  
}  
  
class Div3 extends React.Component<IDivProps> {  
  shouldComponentUpdate(nextProps: IDivProps, nextState: any) {  
    console.log("nextProps", nextProps);  
    console.log("nextState", nextState);  
    if (nextProps.num === this.props.num) {  
      return false;  
    }  
    return true;  
  }  
  render() {  
    console.log("Div3", this.props.num);  
    return <p>{this.props.num}</p>;  
  }  
}  
  
interface IExample4State {  
  num: number;  
}  
  
class Example4 extends React.Component<{}, IExample4State> {  
  constructor(props: any) {  
    super(props);  
    this.state = {  
      num: 0,  
    };  
  }  
  click = () => {  
    this.setState({  
      num: 1,  
    });  
  };  
  render() {  
    return (  
      <>  
        <Div1 num={this.state.num}></Div1>  
        <Div2 num={this.state.num}></Div2>  
        <Div3 num={this.state.num}></Div3>  
  
        <button onClick={this.click}>Click</button>  
      </>  
    );  
  }  
}  
  
class App extends React.Component {  
  render() {  
    return (  
      <>    
        <hr />  
        <Example4 />  
      </>  
    );  
  }  
}  
  
export default App;  