import {sum} from './calc'
import './style.css';

export default function bar(){

  console.log("bar 1");
  const testSTR = "hello world";
  document.body.innerHTML  = `<h1 class="hello">${new Date}</h1><br>${sum(5241,3)}<br />${testSTR}`

  
}