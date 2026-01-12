import React from 'react'

import Comchildren1 from '../../components/Testcomponent'



const Demo2: React.FC<{}> = () => {
  return (
    <div>
      <h1>高阶组件 异步按需加载组件</h1>
      <Comchildren1 />
    </div>
  );
}

export default Demo2