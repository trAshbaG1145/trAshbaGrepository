import React from "react";

import { useParams, useSearchParams, useNavigate } from "react-router-dom";

export default function Comchildren1() {
  const params = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const nav = useNavigate();


  return (
    <div>
      <div>当前展示的组件是：q3</div>
      <div>ID：{params?.id}</div>
      <div>a:{searchParams.get("a")}</div>
      <div>c:{searchParams.get("c")}</div>
      <button
        onClick={() => {
          nav("/QiantaoIndex/q1");
        }}
      >
        go to page q1
      </button>
    </div>
  );
}
