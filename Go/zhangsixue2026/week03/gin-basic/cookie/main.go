package main

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type LoginForm struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 真实场景解密cookie
		username, _ := c.Cookie("username")
		password, _ := c.Cookie("password")
		// 简单验证用户名和密码是否正确，真实场景是否需要查询数据库
		if username != "zhangsan" || password != "123456" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Unauthorized"})
			c.Abort()
			return
		}

		c.Next()
	}
}

func setupRoutes(r *gin.Engine) {
	r.POST("/login", func(c *gin.Context) {
		var form LoginForm
		if err := c.ShouldBindJSON(&form); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if form.Username != "zhangsan" || form.Password != "123456" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid credentials"})
			return
		}
		// 设置cookie，过期时间1小时（！！！注意:真实场景不会把密码放入cookie中，此处只作为示例，jwt会更安全）
		// 真实场景加密处理
		c.SetCookie("username", form.Username, int(time.Hour.Seconds()), "/", "", false, true)
		c.SetCookie("password", form.Password, int(time.Hour.Seconds()), "/", "", false, true)
		c.JSON(http.StatusOK, gin.H{"message": "登录成功"})
	})
	// 用户注销清除Cookie
	r.POST("/logout", func(c *gin.Context) {
		// 删除认证相关的Cookie
		c.SetCookie("username", "", -1, "/", "", false, true)
		c.SetCookie("password", "", -1, "/", "", false, false)
		c.JSON(200, gin.H{
			"message": "注销成功",
		})
	})

	r.GET("/getcookie", func(c *gin.Context) {
		username, err := c.Cookie("username")
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "Cookie not found"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"username": username})
	})
	r.GET("/protected", AuthMiddleware(), func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "Access granted"})
	})
}
func main() {
	// 创建默认的路由引擎
	r := gin.Default()
	r.Static("/static", "./static")
	setupRoutes(r)

	// 启动服务器
	r.Run(":8080")
}
