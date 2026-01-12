const fs = require("fs");
const path = require("path");

// 读取文件内容（同步）
const filePath = path.resolve(__dirname, "example.txt");
const fileContent = fs.readFileSync(filePath, "utf8");
console.log("文件内容：", fileContent);

// 追加文件内容（同步）
const appendContent = "\n这是追加的内容";
fs.appendFileSync(filePath, appendContent);
console.log("文件内容已成功更新");

// fs.writeFileSync(filePath, "这是新的内容");
