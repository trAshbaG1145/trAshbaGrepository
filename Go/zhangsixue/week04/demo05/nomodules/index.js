function domReady() {

  document.getElementById("btn_sum").addEventListener("click", btnSumClick);
}

function btnSumClick() {
  let num1 = parseInt(document.getElementById("num_x").value.trim());
  let num2 = parseInt(document.getElementById("num_y").value.trim());

  let sum_res = sum(num1, num2);
  alert(sum_res);
}
console.log("sum",concat("Hello!","wps"),x,y)

document.addEventListener("DOMContentLoaded", domReady);

