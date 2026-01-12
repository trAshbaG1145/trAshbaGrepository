import React from 'react'
const Comchildren1 = React.lazy(() => import('../../components/Testcomponent'))

const Demo2: React.FC<{}> = () => {
  return (
    <React.Suspense fallback={<div>Loading...</div>}>
      <h1>高阶组件 异步按需加载组件 对比</h1>
      <Comchildren1 />
    </React.Suspense>
  );
}

export default Demo2