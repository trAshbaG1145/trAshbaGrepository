import React from 'react'

import AsyncRouter from '../../utils/AsyncRouter'


const Comchildren2 = AsyncRouter(()=>import('../../components/Testcomponent'))

const Demo2: React.FC<{}> = () => {
  return (
    <div>
      <h1>高阶组件 异步按需加载组件 对比</h1>
      <Comchildren2 />
    </div>
  );
}

export default Demo2