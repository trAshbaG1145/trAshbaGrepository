package main

import (
	"log"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.New()

	r.Use(MiddlewareA())
	r.Use(MiddlewareB())
	r.Use(MiddlewareC())

	r.GET("/example", func(c *gin.Context) {
		log.Println("执行路由处理函数")
		c.JSON(200, gin.H{"message": "成功"})
	})

	r.Run(":8080")
}

func MiddlewareA() gin.HandlerFunc {
	return func(c *gin.Context) {
		log.Println(c.Request.URL.Path)
		log.Println("MiddlewareA: 进入")
		c.Next()
		log.Println("MiddlewareA: 退出")
	}
}

func MiddlewareB() gin.HandlerFunc {

	return func(c *gin.Context) {
		log.Println("MiddlewareB: 进入")
		c.Next()
		log.Println("MiddlewareB: 退出")
	}
}

func MiddlewareC() gin.HandlerFunc {
	return func(c *gin.Context) {
		log.Println("MiddlewareC: 进入")
		c.Next()
		log.Println("MiddlewareC: 退出")

	}
}
