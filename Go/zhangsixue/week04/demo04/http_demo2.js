const http = require("http");
const fs = require("fs");
const { resolve } = require("path");
const hostname = "127.0.0.1";
const port = 3000;

const server = http.createServer((req, res) => {
  const content = fs.readFileSync(resolve(__dirname, "fs_demo.js"), {
    encoding: "utf-8",
  });

  res.statusCode = 200;
  res.setHeader("Content-Type", "application/javascript");
  res.end(content);
});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
