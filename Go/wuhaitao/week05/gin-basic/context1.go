package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// 模拟用户信息结构体
type User struct {
	ID   int
	Name string
}

// 模拟考试信息结构体
type Exam struct {
	ID    int
	Title string
}

// 模拟数据库存储用户信息
var users = map[int]User{
	1: {ID: 1, Name: "Alice"},
	2: {ID: 2, Name: "Bob"},
}

// 模拟数据库存储考试信息，这里假设每个用户有固定的考试列表
var exams = map[int][]Exam{
	1: {
		{ID: 101, Title: "Math Exam"},
		{ID: 102, Title: "English Exam"},
	},
	2: {
		{ID: 201, Title: "Science Exam"},
		{ID: 202, Title: "History Exam"},
	},
}

// 获取用户信息中间件
func getUserInfo(c *gin.Context) {
	userID := 1 // 简单模拟获取用户ID，实际应用中可能从认证等方式获取
	user, exists := users[userID]
	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
		c.Abort()
		return
	}
	c.Set("user", user)
	c.Next()
}

// 获取用户考试列表中间件
func getUserExams() gin.HandlerFunc {
	return func(c *gin.Context) {
		user, exists := c.Get("user")
		if !exists {
			c.JSON(http.StatusBadRequest, gin.H{"error": "User not set in context"})
			c.Abort()
			return
		}
		userObj, ok := user.(User)
		if !ok {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid user data in context"})
			c.Abort()
			return
		}
		userExams, exists := exams[userObj.ID]
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Exams not found for this user"})
			c.Abort()
			return
		}
		// 将考试列表存储到上下文中
		c.Set("exams", userExams)
		c.Next()
	}
}

func main() {
	r := gin.Default()

	r.GET("/user/exams", getUserInfo, getUserExams(), func(c *gin.Context) {
		// 从上下文中获取考试列表
		exams, _ := c.Get("exams")
		c.JSON(http.StatusOK, gin.H{"exams": exams})
	})

	r.Run(":8080")
}
