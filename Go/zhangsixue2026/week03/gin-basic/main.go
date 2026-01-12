package main

import "github.com/gin-gonic/gin"

func main() {
	// 创建默认的路由引擎
	r := gin.Default()

	// 定义一个GET路由
	r.GET("/hello", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "Hello, Gin!",
		})
	})

	// 启动服务器
	r.Run(":8080")
}
