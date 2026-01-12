package main

import (
	"github.com/gin-gonic/gin"
)

func main() {
	// 创建默认的路由引擎
	r := gin.Default()
	r.Static("/static", "./test")

	// 启动服务器
	r.Run(":8080")
}
