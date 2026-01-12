package main

import (
	"fmt"
	"log"
	"time"

	"github.com/gin-gonic/gin"
)

func Logger() gin.HandlerFunc {
	return func(c *gin.Context) {
		t := time.Now()

		// 设置一个变量，可以被后续的处理函数访问
		c.Set("example", "这是一个来自中间件的值")

		fmt.Println("1")

		// 继续执行后续的中间件和路由处理函数
		c.Next()
		fmt.Println("2")
		// 计算请求处理耗时
		latency := time.Since(t)

		// 打印路径、状态码和耗时
		log.Printf("[%s] %s %d %v", c.Request.Method, c.Request.URL.Path, c.Writer.Status(), latency)
	}
}

func main() {
	r := gin.New() // 不使用默认中间件

	// 使用自定义Logger中间件
	r.Use(Logger())

	r.GET("/test", func(c *gin.Context) {
		// 获取中间件设置的值
		fmt.Println("3")
		example := c.MustGet("example").(string)
		c.JSON(200, gin.H{
			"message":          "测试成功",
			"middleware_value": example,
		})
	})

	r.Run(":8080")
}
