import React, { useRef } from 'react';
import { Outlet, Link } from "react-router-dom";

function App() {

  const ID = Math.random().toString(36).slice(-8);
  const ULR = `/QiantaoIndex/q3/${ID}?a=b&c=d`;
  return (
    <div>
      <div>嵌套路由demo </div>
      <Link to="/QiantaoIndex/q1">q1</Link> | <Link to="/QiantaoIndex/q2">q2</Link> |<Link to={ULR}>q3</Link>
      <Outlet />
    </div>
  );
}

export default App;
