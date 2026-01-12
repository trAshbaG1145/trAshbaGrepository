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

	// 获取查询参数
	r.GET("/search", func(c *gin.Context) {
		query := c.Query("q")               // 获取参数，没有则返回空字符串
		page := c.DefaultQuery("page", "1") // 获取参数，没有则使用默认值

		c.JSON(200, gin.H{
			"query": query,
			"page":  page,
		})
	})

	// 启动服务器
	r.Run(":8084")
}
