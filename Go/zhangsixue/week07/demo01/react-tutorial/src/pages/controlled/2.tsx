import React, { useRef, useEffect } from "react";

function query(value: string) {
  fetch("/api/data?pangeNo=2&pageSize=510&name=" + value)
    .then((res) => res.json())
    .then((res) => console.log(res));
}
// 非受控组件
function UncontrolledInput() {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    // alert(`Input value: ${inputRef?.current?.value}`);
    const value = inputRef?.current?.value
    if (value) {
      query(inputRef?.current?.value);
    }
  };
  console.log("render UncontrolledInput");

  return (
    <div>
      <input type="text" ref={inputRef} defaultValue="Default value" />
      <button onClick={handleClick}>Show Input Value</button>
    </div>
  );
}

// 受控组件
function ControlledInput() {
  const [value, setValue] = React.useState("");

  useEffect(() => {
    if (value) {
      // 防抖节流
      query(value);
    }
  }, [value]);
  console.log("render ControlledInput");

  return (
    <div>
      <input
        type="text"
        onChange={(e) => setValue(e.target.value)}
        defaultValue="Default value"
      />
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
      <UncontrolledInput />
      <hr />
      <ControlledInput />
    </div>
  );
};
