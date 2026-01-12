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

    this.setState(
      {
        dataList: originList,
      },
      () => {
        this.toRenderList();
      }
    );
  };

  toRenderList = () => {
    const { renderList } = this.state;
    renderList.push(this.renderNewList());
    this.setState({
      renderList,
    });
  };

  renderNewList() {
    /* 得到最新的渲染列表 */
    const { dataList } = this.state;
    const list = dataList;
    return (
      <React.Fragment>
        {list.map((_, listIndex) => (
          <Circle key={_} value={_ + ""} />
        ))}
      </React.Fragment>
    );
  }

  render() {
    return (
      <>
        <h1>未做时间分片</h1>
        <button onClick={this.handleClick}>展示效果</button>
        <div style={{ width: "100%", position: "relative" }}>
          {this.state.renderList}
        </div>
      </>
    );
  }
}
