const http = require("http");

const server = http.createServer((req, res) => {
  res.setHeader("Content-Type", "application/json");
  // res.setHeader('Access-Control-Allow-Origin', '*');
  const url = req.url;
  console.log("url", url);
  if (url == "/api/test") {
    const data = {
      code: 0,
      list: [
        { id: 1, name: "John Doe1" },
        { id: 2, name: "Jane Doe2" },
        { id: 3, name: "Jim Doe3" },
      ],
    };

    res.end(JSON.stringify(data));
  } else {
    res.writeHead(404);
    res.end(JSON.stringify({ error: "Not Found" }));
    return;
  }
});

const PORT = 8000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
