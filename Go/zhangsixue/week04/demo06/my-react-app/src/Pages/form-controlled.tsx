import React, { useRef, useEffect } from "react";

// 非受控组件
function UncontrolledInput() {
  const inputRef = useRef<HTMLInputElement>(null);
// debugger;
  const handleClick = () => {
    debugger;
    alert(`Input value: ${inputRef?.current?.value}`);
  };
  console.log('render UncontrolledInput')
  return (
    <div>
      <input type="text" ref={inputRef} defaultValue="wh" />
      <button onClick={handleClick}>Show Input Value</button>
    </div>
  );
}

// 受控组件
function ControlledInput() {
  const [value, setValue] = React.useState("518");

  const handleClick = () => {
    alert(value);
  };
  console.log('render ControlledInput',Math.random())
  return (
    <div>
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        // defaultValue="Default value"
      />
      <button onClick={handleClick}>Show Input Value</button>
    </div>
  );
}
export default () => {
  return (
    <div>
      <UncontrolledInput />
      <hr />
      <ControlledInput />
    </div>
  );
};
