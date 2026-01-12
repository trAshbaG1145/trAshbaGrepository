import React,{useEffect} from "react";
import './index.css'

//添加样式
//添加缓存
function App() {
  const [todos, setTodos] = React.useState([]);
  const [input, setInput] = React.useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input === "") return;
    setTodos([...todos, { id: Date.now(), text: input, completed: false }]);
    setInput("");
  };

  useEffect(() => {
    fetch("/api/test").then(res => res.json()).then(data => {
      console(data);
    }
    ).catch(err => {
      console.log(err);
    })
  }, [])

  return (
    <div className="App">
      <h1>Todo List123</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}

        />
      </form>
      <ul>
        {
          todos.map(v => {
            return (
              <li key={v.id}>
                <input type="checkbox" checked={v.completed} onChange={() => {
                  setTodos(todos.map(todo => todo.id === v.id ? { ...todo, completed: !todo.completed } : todo))
                }} />
                {v.text}
                <button onClick={() => {
                  setTodos(todos.filter(todo => todo.id !== v.id))
                }}>删除</button>
              </li>
            )
          })
        }
      </ul>

    </div>
  );
}

export default App;
