import React  from "react";  


interface IExample2State {  
  count: number;  
}  
  
// 使用React.Component时，state没有变化时，只要内部调用setState，则组件重新渲染（hook会有优化，不会这样）
// 优化方式两种：
// 1. 使用React.PureComponent，state没有变化时，不会重新渲染
// 2. 使用shouldComponentUpdate，state没有变化时，不会重新渲染
class Example1 extends React.Component<{}, IExample2State> {  
  state = { count: 1 };  
  
  componentDidMount() {  
    console.log("componentDidMount ");  
  }  
  
  componentWillUnmount() {  
    console.log("componentWillUnmount ");  

  }  
  // shouldComponentUpdate(nextProps: {}, nextState: IExample2State) {  
  //   console.log("shouldComponentUpdate nextProps,nextState", nextProps, nextState);  
  //   return true;  
  // } 
  
  tick = () => {  
    this.setState({  
      count: 1,  
    });  
  }  
  
  render() {  
    const {count} = this.state
    console.log('render')
    return (  
      <div>  
        <div>Hello, world!</div>  
        <div>It is {count}</div>  
        <button onClick={this.tick}>Tick</button>
      </div>  
    );  
  }  
}  

export default Example1