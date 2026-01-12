package main

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/dgrijalva/jwt-go"
	"github.com/gin-gonic/gin"
)

// 定义密钥，在实际应用中应该使用更安全的密钥管理方式
const secretKey = "basdjj3424133232"

// GenerateToken 生成模拟JWT
func GenerateToken(userID string) (string, error) {
	// 创建一个新的JWT对象
	token := jwt.New(jwt.SigningMethodHS256)

	// 设置JWT的声明（Claims），这里包含用户ID和过期时间
	claims := token.Claims.(jwt.MapClaims)
	claims["user_id"] = userID
	claims["exp"] = time.Now().Add(time.Hour * 1).Unix() // 1小时后过期

	// 使用密钥对JWT进行签名
	signedToken, err := token.SignedString([]byte(secretKey))
	if err != nil {
		return "", err
	}

	return signedToken, nil
}

// VerifyToken 验证和解密模拟JWT
func VerifyToken(tokenStr string) (string, error) {
	token, err := jwt.Parse(tokenStr, func(token *jwt.Token) (interface{}, error) {
		// 验证签名方法
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(secretKey), nil
	})

	if err != nil {
		return "", err
	}

	if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
		// 检查令牌是否过期
		if float64(time.Now().Unix()) > claims["exp"].(float64) {
			return "", errors.New("token has expired")
		}

		// 获取用户ID
		userID, ok := claims["user_id"].(string)
		if !ok {
			return "", errors.New("user_id claim not found or invalid")
		}

		return userID, nil
	}

	return "", errors.New("invalid token")
}

type LoginForm struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 真实场景解密cookie
		token, _ := c.Cookie("token")
		userID, err := VerifyToken(token)
		// 数据库验证用户名和密码
		if userID != "zhangsan" || err != nil {
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
		// 数据库验证用户名和密码
		if form.Username != "zhangsan" || form.Password != "123456" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid credentials"})
			return
		}
		token, err := GenerateToken(form.Username)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate token"})
			return
		}
		c.SetCookie("token", token, int(time.Hour.Seconds()), "/", "", false, true)
		c.JSON(http.StatusOK, gin.H{"message": "登录成功"})
	})
	// 用户注销清除Cookie
	r.POST("/logout", func(c *gin.Context) {
		// 删除认证相关的Cookie
		c.SetCookie("token", "", -1, "/", "", false, true)
		c.JSON(200, gin.H{
			"message": "注销成功",
		})
	})
	r.GET("/getcookie", func(c *gin.Context) {
		token, err := c.Cookie("token")
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "Cookie not found"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"token": token})
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
