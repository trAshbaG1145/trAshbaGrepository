import React, { useState, useCallback } from "react";

export default function Comchildren1() {
  const [count, setCount] = useState(0);

  const add = useCallback(() => {
    setCount((count) => count + 1);
  }, []);

  return (
    <div>
      <h1>Comchildren1</h1>
      <p>{count}</p>
      <button onClick={add}>确定</button>
    </div>
  );
}
