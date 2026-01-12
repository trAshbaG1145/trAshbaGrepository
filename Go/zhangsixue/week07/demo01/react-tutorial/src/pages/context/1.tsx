import React, { useState, useContext } from "react";

type Theme = "Light" | "Dark";

interface ThemeContextType {
  theme: Theme;
  toggle?: () => void;
}

const ThemeContext = React.createContext<ThemeContextType>({ theme: "Light" });

interface Iprops {
  children?: React.ReactNode;
}
const App: React.FC<Iprops> = (props) => {
  const { children } = props;
  const [theme, setTheme] = useState<Theme>("Light");

  const toggle = () => {
    console.log("toggle");
    setTheme(theme === "Light" ? "Dark" : "Light");
  };
  console.log("theme:", theme);
  return (
    <ThemeContext.Provider value={{ theme, toggle }}>
      <Comchildren1 />
    </ThemeContext.Provider>
  );
};

function Comchildren1() {
  return (
    <div>
      <h1>Context 使用</h1>
      <Comchildren2 />
      <Comchildren3 />
    </div>
  );
}

// 推荐使用方式 hook
function Comchildren2() {
  const context = useContext(ThemeContext);
  const { theme, toggle } = context;
  console.log("context", context);
  return (
    <div>
      <p>Current theme: {theme}</p>
      <button type="button" onClick={toggle}>
        toggle theme
      </button>
    </div>
  );
}


// Consumer 方式
function Comchildren3() {
  return (
    <ThemeContext.Consumer>
      {({ theme, toggle }) => (
        <div>
          <p>Current theme: {theme}</p>
          <button type="button" onClick={toggle}>
            toggle theme
          </button>
        </div>
      )}
    </ThemeContext.Consumer>
  );
}

export default App;
