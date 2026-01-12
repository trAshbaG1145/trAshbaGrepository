package main

import (
	"net/http"
	"strconv"
	"sync"

	"github.com/gin-gonic/gin"
)

// 模拟用户信息结构体
type User struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}

// 创建用户请求体
type CreateUserRequest struct {
	Name string `json:"name" binding:"required"`
}

// 模拟数据库存储用户信息
var users = map[int]User{
	1: {ID: 1, Name: "Alice"},
	2: {ID: 2, Name: "Bob"},
}

// 用于生成用户ID的计数器
var userIDCounter = 3

// 互斥锁，保证并发安全
var usersMutex sync.RWMutex

// 辅助函数：获取下一个用户ID
func getNextUserID() int {
	userIDCounter++
	return userIDCounter - 1
}

func main() {
	r := gin.Default()

	// 用户相关的RESTful接口
	// 获取所有用户 GET /users
	r.GET("/users", func(c *gin.Context) {
		usersMutex.RLock()
		defer usersMutex.RUnlock()

		userList := make([]User, 0, len(users))
		for _, user := range users {
			userList = append(userList, user)
		}

		c.JSON(http.StatusOK, gin.H{
			"code":    200,
			"message": "获取用户列表成功",
			"data":    userList,
			"total":   len(userList),
		})
	})

	// 获取单个用户 GET /users/:id
	r.GET("/users/:id", func(c *gin.Context) {
		idStr := c.Param("id")
		id, err := strconv.Atoi(idStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"code":    400,
				"message": "无效的用户ID",
			})
			return
		}

		usersMutex.RLock()
		user, exists := users[id]
		usersMutex.RUnlock()

		if !exists {
			c.JSON(http.StatusNotFound, gin.H{
				"code":    404,
				"message": "用户不存在",
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"code":    200,
			"message": "获取用户成功",
			"data":    user,
		})
	})

	// 创建单个用户 POST /users
	r.POST("/users", func(c *gin.Context) {
		var req CreateUserRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"code":    400,
				"message": "请求参数错误",
				"error":   err.Error(),
			})
			return
		}

		usersMutex.Lock()
		newUser := User{
			ID:   getNextUserID(),
			Name: req.Name,
		}
		users[newUser.ID] = newUser
		usersMutex.Unlock()

		c.JSON(http.StatusCreated, gin.H{
			"code":    201,
			"message": "用户创建成功",
			"data":    newUser,
		})
	})
	// 更新单个用户 PUT /users/:id
	r.PUT("/users/:id", func(c *gin.Context) {
		idStr := c.Param("id")
		id, err := strconv.Atoi(idStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"code":    400,
				"message": "无效的用户ID",
			})
			return
		}

		var req CreateUserRequest // 复用创建请求的结构体
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"code":    400,
				"message": "请求参数错误",
				"error":   err.Error(),
			})
			return
		}

		usersMutex.Lock()
		_, exists := users[id]
		if !exists {
			usersMutex.Unlock()
			c.JSON(http.StatusNotFound, gin.H{
				"code":    404,
				"message": "用户不存在",
			})
			return
		}

		updatedUser := User{
			ID:   id,
			Name: req.Name,
		}
		users[id] = updatedUser
		usersMutex.Unlock()

		c.JSON(http.StatusOK, gin.H{
			"code":    200,
			"message": "用户更新成功",
			"data":    updatedUser,
		})
	})

	// 删除单个用户 DELETE /users/:id
	r.DELETE("/users/:id", func(c *gin.Context) {
		idStr := c.Param("id")
		id, err := strconv.Atoi(idStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"code":    400,
				"message": "无效的用户ID",
			})
			return
		}

		usersMutex.Lock()
		user, exists := users[id]
		if !exists {
			usersMutex.Unlock()
			c.JSON(http.StatusNotFound, gin.H{
				"code":    404,
				"message": "用户不存在",
			})
			return
		}

		delete(users, id)
		usersMutex.Unlock()

		c.JSON(http.StatusOK, gin.H{
			"code":    200,
			"message": "用户删除成功",
			"data":    user,
		})
	})

	r.Run(":8080")
}
