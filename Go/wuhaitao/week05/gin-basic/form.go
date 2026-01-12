package main

import (
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
)

// LoginForm 定义了登录表单的结构体
type LoginForm struct {
	Username string `form:"username" binding:"required"`       // 从表单的 'username' 字段获取，且必填
	Password string `form:"password" binding:"required,min=6"` // 从 'password' 字段获取，必填且最小长度为6
	Remember bool   `form:"remember"`                          // 可选的复选框字段
}

func main() {
	router := gin.Default()

	// 处理 /login POST 请求
	router.POST("/login/form", func(c *gin.Context) {
		// 1. 定义一个 LoginForm 类型的变量
		var form LoginForm

		// 2. 使用 ShouldBind 尝试绑定表单数据
		// ShouldBind 会根据 Content-Type 自动选择合适的绑定器
		// 对于 application/x-www-form-urlencoded 或 multipart/form-data，它会解析表单数据

		if err := c.ShouldBind(&form); err != nil {
			log.Printf("绑定表单失败: %v\n", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// 3. 绑定成功，可以访问 form 结构体中的数据
		log.Printf("成功接收到登录表单数据: Username=%s, Password=%s, Remember=%t\n", form.Username, form.Password, form.Remember)

		// 模拟登录验证逻辑...
		if form.Username == "admin" && form.Password == "admin123" {
			c.JSON(http.StatusOK, gin.H{"status": "登录成功"})
		} else {
			c.JSON(http.StatusUnauthorized, gin.H{"status": "用户名或密码错误"})
		}
	})

	log.Println("服务器启动，监听端口 :8080")
	router.Run(":8080")
}
