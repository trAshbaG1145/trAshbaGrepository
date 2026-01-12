import React, { useEffect, useState } from "react";

const Demo = () => {
  const [data, setData] = useState<any>([]); // 所有数据
  const [visibleData, setVisibleData] = useState([]); // 可见数据
  const [scrollTop, setScrollTop] = useState(0); // 滚动位置

  const itemHeight = 20; // 每项高度
  const containerHeight = 300; // 容器高度
  const bufferSize = 5; // 缓冲区大小

  useEffect(() => {
    // 模拟获取数据
    const fetchData = () => {
      const allData = Array.from({ length: 40000 }, (_, index) => index + 1);
      setData(allData);
    };

    fetchData();
  }, []);

  useEffect(() => {
    console.log("render");
    // 计算可见数据
    const startIndex = Math.floor(scrollTop / itemHeight);
    const endIndex = Math.min(
      startIndex + Math.ceil(containerHeight / itemHeight) + bufferSize,
      data.length
    );
    setVisibleData(data.slice(startIndex, endIndex));
  }, [scrollTop, data]);

  const handleScroll = (event: React.WheelEvent<HTMLDivElement>) => {
    setScrollTop(event.currentTarget.scrollTop);
  };

  return (
    <div
      style={{ height: `${containerHeight}px`, overflow: "auto" }}
      onScroll={handleScroll}
    >
      <h1>虚拟列表</h1>
      <div
        style={{
          height: `${data.length * itemHeight}px`,
          paddingTop: `${scrollTop}px`,
        }}
      >
        {visibleData.map((item) => (
          <div key={item} style={{ height: `${itemHeight}px` }}>
            {item}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Demo;
