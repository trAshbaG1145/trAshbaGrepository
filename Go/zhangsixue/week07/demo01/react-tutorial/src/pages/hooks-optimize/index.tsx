import React, { useState, useCallback, memo, useMemo } from "react";

interface Iprops {
  onClick?: () => void;
  children?: React.ReactNode;
}
function Button2(props: Iprops) {
  console.log("Button2 render");
  return <button onClick={props.onClick}>{props.children}</button>;
}
const Button3 = memo(Button2);
const Button4 = memo(Button2, (prevProps, nextProps) => {
  return prevProps.onClick === nextProps.onClick;
});

function Example1() {
  const [num, setNum] = useState(1);

  const addOne = useCallback(() => {
    setNum((num) => num + 1);
  }, []);
  return (
    <>
      <div>{num}</div>
      <Button2 onClick={addOne}>Add</Button2>
    </>
  );
}

function Example2() {
  const [num, setNum] = useState(1);
  console.time("Example2 addOne");
  const addOne = useCallback(() => {
    setNum((num) => num + 1);
  }, []);
  console.timeEnd("Example2 addOne");
  return (
    <>
      <div>{num}</div>
      <Button3 onClick={addOne}>add</Button3>
    </>
  );
}

function Example3() {
  const [num, setNum] = useState(1);

  console.time("Example3 addOne");
  const addOne = () => {
    setNum((num) => num + 1);
  };
  console.timeEnd("Example3 addOne");
  return (
    <>
      <div>{num}</div>
      <Button3 onClick={addOne}>Add</Button3>
    </>
  );
}

function Example2_1() {
  const [num, setNum] = useState(1);
  const addOne = useCallback(() => {
    setNum((num) => num + 1);
  }, []);
  return (
    <>
      <div>{num}</div>
      <Button3 onClick={addOne}>
        <Example2_1_1>ADD</Example2_1_1>
      </Button3>
    </>
  );
}
function Example2_2() {
  const [num, setNum] = useState(1);
  const addOne = useCallback(() => {
    setNum((num) => num + 1);
  }, []);
  return (
    <>
      <div>{num}</div>
      <Button4 onClick={addOne}>
        <Example2_1_1>ADD</Example2_1_1>
      </Button4>
    </>
  );
}

function Example2_1_1(props: Iprops) {
  console.log("Example2_1_1");
  return <div>{props.children}</div>;
}

function Example4() {
  const [num, setNum] = useState(1);

  const addOne = useCallback(() => {
    setNum((num) => num + 1);
  }, []);
  return (
    <>
      <div>{num}</div>
      {useMemo(
        () => (
          <Button2 onClick={addOne}>Add</Button2>
        ),
        []
      )}
    </>
  );
}

export default function App() {
  return (
    <div>
      <h1>useCallback & useMemo</h1>
      <Example1 />
      <hr />
      <Example2 />
      <div style={{ padding: "10px" }}>
        <Example2_1 />
        <Example2_2 />
      </div>
      <hr />
      <Example3 />
      <hr />
      <Example4 />
    </div>
  );
}
