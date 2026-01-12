// 时间分片
import React, { CSSProperties } from "react";

interface StateTypes {
  renderList: JSX.Element[];
  dataList: number[];
  eachRenderNum: number;
}

/* 色块组件 */
function Circle({ value }: { value: string }) {
  const style: CSSProperties = {
    width: "20px",
    height: "20px",
    padding: 0,
    margin: "5px 0",
  };
  return <p style={style}>{value}</p>;
}

export default class Demo extends React.Component {
  state: StateTypes = {
    dataList: [], //数据源列表
    renderList: [], //渲染列表
    eachRenderNum: 100, // 每次渲染数量
  };

  handleClick = () => {
    const originList = Array.from({ length: 40000 }, (_, index) => index + 1);
    // 计算需要渲染次数
    const times = Math.ceil(originList.length / this.state.eachRenderNum);
    let index = 1;
    this.setState(
      {
        dataList: originList,
      },
      () => {
        this.toRenderList(index, times);
      }
    );
  };

  toRenderList = (index: number, times: number) => {
    if (index > times) return; /* 如果渲染完成，那么退出 */
    const { renderList } = this.state;
    renderList.push(this.renderNewList(index));
    this.setState({
      renderList,
    });
    // // 浏览器空闲执行下一批渲染
    requestIdleCallback(() => {
      this.toRenderList(++index, times);
    });
  };

  renderNewList(index: number) {
    /* 得到最新的渲染列表 */
    const { dataList, eachRenderNum } = this.state;
    const list = dataList.slice(
      (index - 1) * eachRenderNum,
      index * eachRenderNum
    );
    return (
      <React.Fragment key={index}>
        {list.map((_, listIndex) => (
          <Circle key={_} value={_ + ""} />
        ))}
      </React.Fragment>
    );
  }

  render() {
    return (
      <>
        <h1>时间分片</h1>
        <button onClick={this.handleClick}>展示效果</button>
        <div style={{ width: "100%", position: "relative" }}>
          {this.state.renderList}
        </div>
      </>
    );
  }
}
