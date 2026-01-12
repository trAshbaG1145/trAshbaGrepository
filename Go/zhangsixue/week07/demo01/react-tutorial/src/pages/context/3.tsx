
import React, { useState, useContext } from 'react';


type Theme = 'Light' | 'Dark';

interface ThemeContextType {
  theme: Theme;
  toggle?: () => void;
}

const ThemeContext = React.createContext<ThemeContextType>({ theme: "Light" });

interface Iprops {
  children?: React.ReactNode;
}
const  ThemeProvider:React.FC<Iprops> = (props) =>{
  const { children } = props;
  const [theme, setTheme] = useState<Theme>("Light");

  const toggle = () => {
    setTheme(theme === "Light" ? "Dark" : "Light");
  };
  return (
    <ThemeContext.Provider value={{ theme, toggle }}>
      <Comchildren1 />
      <Comchildren2 />
    </ThemeContext.Provider>
  );
}

function Comchildren1() {
  const context = useContext(ThemeContext);
  const { theme, toggle } = context;
  console.log('App theme:',context)
  return (
    <div>
      <p>Current theme: {theme}</p>
        <button type="button" onClick={toggle}>toggle theme</button>
    </div>
  )
}

function Comchildren2() {
  console.log('this is component2')
    return (
      <div>
        Comchildren2
      </div>
    )
  }
function App() {
  return (
    <>
        <h1>Context 利用children防止子组件渲染 对比</h1>
        <ThemeProvider />
    </>
  );
}

export default App;