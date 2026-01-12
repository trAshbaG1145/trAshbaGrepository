import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { RouterProvider } from "react-router-dom"
// import App from './Pages/todolist.jsx'
import router from "./router"
const root = createRoot(document.getElementById("root"))
root.render(<RouterProvider router={router} />)
// createRoot(document.getElementById('root')).render(
//   // <StrictMode>
//     <App />
//   // </StrictMode>,
// )
