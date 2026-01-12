import React, { useEffect,useState} from "react";

function Page1() {
  const element = <div>Hello, World!</div>;
  const element2 =  document.createElement('div')
  console.log('element',element)
  console.log('element2',element2)

  function Greeting(props) {
    return <div>Hello, <p>{props.name}</p>!</div>;
  }
  console.log('element3',<Greeting name="World" />)

 

  return (
    <div className="App">
      <header className="App-header">
        <div>Hello world</div>
      </header>
    </div>
  );
}

export default Page1;
