package main

import (
	"log"
	"net/http"
)

func main() {
	// 将当前目录作为静态文件服务器的根目录
	fileServer := http.FileServer(http.Dir("."))

	// 注册处理器
	http.Handle("/", fileServer)

	log.Println("文件服务器启动在 :8080...")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
