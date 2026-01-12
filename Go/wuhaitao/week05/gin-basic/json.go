package main

import (
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
)

// User 定义了用户信息的结构体
type User struct {
	Username string `json:"username" binding:"required"`    // binding:"required" 表示该字段是必需的
	Email    string `json:"email" binding:"required,email"` // 必需，且格式需为邮箱
	Age      int    `json:"age" binding:"gte=0,lte=130"`    // 年龄大于等于0，小于等于130
}

type User2 struct {
	Username string `json:"username"` // binding:"required" 表示该字段是必需的
	Email    string `json:"email"`    // 必需，且格式需为邮箱
	Age      int    `json:"age"`      // 年龄大于等于0，小于等于130
}

func main() {
	router := gin.Default()

	// 处理 /register POST 请求
	router.POST("/register/json", func(c *gin.Context) {
		// 1. 定义一个 User 类型的变量
		var user User

		// 2. 使用 ShouldBindJSON 将请求 Body 中的 JSON 绑定到 user 变量
		// ShouldBindJSON 会根据 Content-Type 推断使用 JSON 解码器
		// 如果绑定失败（例如，JSON格式错误、缺少必需字段、类型不匹配），会返回错误
		if err := c.ShouldBindJSON(&user); err != nil {
			// 如果绑定出错，返回 400 Bad Request 错误，并将错误信息返回给客户端
			log.Printf("绑定 JSON 失败: %v\n", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return // 记得 return，防止后续代码执行
		}

		// 3. 绑定成功，可以访问 user 结构体中的数据
		log.Printf("成功接收到用户数据: %+v\n", user) // %+v 会打印字段名和值

		// 4. 返回成功响应
		c.JSON(http.StatusOK, gin.H{
			"message": "用户注册成功!",
			"user":    user,
		})
	})

	log.Println("服务器启动，监听端口 :8080")
	router.Run(":8080")
}
