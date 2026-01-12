import { Outlet, Link, NavLink } from "react-router-dom";

import { routerPath } from "./router"
import './App.css'

function App() {

  return (
    <div>
      <h2>这是首页</h2>
      <div className='container'>
        <ul className='nav'>
          {
            routerPath.map(v => {
              return (
                <li key={v.path}>
                  <NavLink to={v.path} className={({ isActive, isPending }) =>
                    isPending ? "pending" : isActive ? "active" : ""
                  }>{v.path}</NavLink>
                  {/* {
                    v.children && v.children.length > 0 && (
                      <ul>
                        {
                          v.children.map(v => {
                            return (
                              <li key={v.path}>
                                <NavLink to={v.path} className={({ isActive, isPending }) =>
                                  isPending ? "pending" : isActive ? "active" : ""
                                }>{v.path}</NavLink>
                              </li>
                            )
                          })
                        }
                      </ul>
                    )
                  } */}
                </li>
              )
            })
          }
          {/* <li>
          <Link to="Begin">Begin</Link>
        </li> */}
        </ul>
        <div className="main">
          <Outlet />
        </div>
      </div>
    </div>
  )
}

export default App
