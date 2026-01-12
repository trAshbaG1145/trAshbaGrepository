/**
 * 创建一个基本的HTTP服务器并监听指定的主机名和端口号。
 * @module http_basic
 */

const http = require("http");

const hostname = "127.0.0.1";
const port = 3000;

/**
 * 创建HTTP服务器的回调函数。
 * @callback requestCallback
 * @param {http.IncomingMessage} req - 表示客户端请求的对象。
 * @param {http.ServerResponse} res - 表示服务器响应的对象。
 */

/**
 * 创建一个基本的HTTP服务器。
 * @param {requestCallback} requestListener - 处理客户端请求的回调函数。
 * @returns {http.Server} 返回一个HTTP服务器实例。
 */
const server = http.createServer((req, res) => {
  res.statusCode = 200;
  res.setHeader("Content-Type", "text/html");
  res.end("Hello Node.js" + __filename);
  // res.end("Hello, Node.js");
});

/**
 * 启动HTTP服务器并监听指定的主机名和端口号。
 * @param {number} port - 监听的端口号。
 * @param {string} hostname - 监听的主机名。
 * @param {Function} [callback] - 服务器启动后的回调函数。
 */
server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
