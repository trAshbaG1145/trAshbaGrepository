import React from "react";  

// jsx标签语法既不是字符串也不是 HTML
// 在 JSX 中嵌入表达式
function HelloWorld() {
  const name = <span>Hello, world!</span>;
  const element = <h1>Hello, {name}</h1>;
  return <div>{element}</div>
}

function HelloMessage(props) {
  return <div>hello {props.name}</div>;
}
function Greeting() {
  const formatName = (user) => {
    return `${user.firstName} ${user.lastName}`;
  };

  const getGreeting = (user) => {
    if (user) {
      return <h1>Hello, {formatName(user)}!</h1>;
    }
    return <h1>Hello, Stranger!</h1>;
  };

  const user = {
    firstName: "Jack",
    lastName: "Lee",
  };

  return (
    <div>
      <h1>Hello {formatName(user)}!</h1>
      {getGreeting()}
      {getGreeting(user)}
    </div>
  );
}
function Counter() {
  // 数组结构
  const [count, setCount] = React.useState(0);

  const increase = () => {
    setCount(count + 1);
  };

  return (
    <div>
      <span>{count}</span>
      <button onClick={increase}>increase</button>
    </div>
  );
}
function Condition() {
  const count = 0;
  const finished = false;
  const renderText = () => {
    if (count < 10) {
      return <span>小于10</span>;
    } else if (count < 20) {
      return <span>小于20</span>;
    } else {
      return <span>大于等于20</span>;
    }
  };
  return (
    <div>
      <p>status: {finished && <span>finished</span>}</p>
      <p>
        status:{" "}
        {finished ? <span>finished</span> : <span>not finished</span>}
      </p>
      <p>status: {count && <span>finished123</span>}</p>
      <p>
        status:{" "}
        {count ? <span>finished</span> : <span>not finished</span>}
      </p>
      <p>count: {renderText()}</p>
    </div>
  );
}
function List() {
  const list = [1, 2, 3, 4, 5];
  return (
    <div>
      <ul>
        {list.map((item) => (
          <li>{item}</li>
        ))}
      </ul>
      <ul>
        {list.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
function EventHandle() {
  const [areas, setAreas] = React.useState([
    { id: 1, name: "内地" },
    { id: 2, name: "港台" },
    { id: 3, name: "欧美" },
  ]);
  const delItem = (area) => {
    // const index = areas.indexOf(area);
    // areas.splice(index, 1);
    const newAreas = areas.filter((item) => item.id !== area.id);
    setAreas(newAreas);
  };
  return (
    <ul>
      {areas.map((item) => (
        <li key={item.id}>
          {item.name}-<button onClick={() => delItem(item)}>删除</button>
        </li>
      ))}
    </ul>
  );
}
function Input() {
  const [value, setValue] = React.useState("");
  console.log('input',Math.random())
  return (
    <div>
      <input
        type="text"
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
        }}
      />
      <span>{value}</span>
      <button
        onClick={() => {
          setValue("");
        }}
      >
        clear
      </button>
    </div>
  );
}
function Welcome(props) {
  return <div>Hello, {props.children}</div>;
}
function App() {
  return (
    <div>
      <HelloWorld />
      <HelloMessage name="react" />
      {/** 等价于 **/}
      <HelloMessage name="hust"></HelloMessage>
      <Greeting />
      <Counter></Counter>
      <Condition />
      <List />
      <EventHandle />
      <Input />
      <Welcome>
        <span>wps</span>
      </Welcome>
    </div>
  );
}

export default App;
