import { createHashRouter, createBrowserRouter } from "react-router-dom"
import Begin from "./Pages/begin"
import Events from "./Pages/events"
import Formcontrolled from "./Pages/form-controlled"
import Hooksstateeffect from "./Pages/hooks-state-effect"
import Lifecycle from "./Pages/life-cycle"
import Syntax1 from "./Pages/syntax-1"
import Syntax2 from "./Pages/syntax-2"
import Todolist from "./Pages/todolist"

import QiantaoIndex from "./Pages/router/qIndex"
import Q1 from "./Pages/router/q1"
import Q2 from "./Pages/router/q2"
import Q3 from "./Pages/router/q3"

import NotFound from "./Pages/NotFound"
import App from "./App"


export const routerPath = [
  {
    path: "begin",
    element: <Begin />,
  },
  {
    path: "events",
    element: <Events />,
  },
  {
    path: "formcontrolled",
    element: <Formcontrolled />,
  },
  {
    path: "hooksstateeffect",
    element: <Hooksstateeffect />,
  },

  {
    path: "lifecycle",
    element: <Lifecycle />,
  },
  {
    path: "syntax1",
    element: <Syntax1 />,
  },
  {
    path: "syntax2",
    element: <Syntax2 />,
  },
  {
    path: "todolist",
    element: <Todolist />,
  },
  {
    path: "qiantaoIndex",
    element: <QiantaoIndex />,
    children: [
      {
        index: true,
        element: <Q1 />,
      },
      {
        path: "q1",
        element: <Q1 />,
      },
      {
        path: "q2",
        element: <Q2 />,
      },
      {
        path: "q3/:id?",
        element: <Q3 />,
      },
    ]
  }
]

const router = createHashRouter([
  {
    path: "/",
    element: <App />,
    errorElement: <NotFound />,
    children: routerPath,
  },

])

export default router
