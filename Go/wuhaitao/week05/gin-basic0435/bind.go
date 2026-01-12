package main

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
)

// User 定义用户结构体
type User struct {
	Name string `json:"name,omitempty"`
	Age  int    `json:"age,omitempty"`
}

func main() {
	r := gin.Default()

	r.POST("/bindjson", func(c *gin.Context) {
		var user User
		c.BindJSON(&user)
		c.JSON(http.StatusOK, gin.H{"user": user})
	})

	r.POST("/shouldbindjson", func(c *gin.Context) {
		var user User
		if err := c.ShouldBindJSON(&user); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"user": user})
	})

	r.POST("/shouldbind", func(c *gin.Context) {
		var user User
		if err := c.ShouldBind(&user); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		fmt.Println("user", user)
		c.JSON(http.StatusOK, gin.H{"user": user})
	})

	r.POST("/form", func(c *gin.Context) {

		name := c.PostForm("name")
		age := c.PostForm("age")
		c.JSON(http.StatusOK, gin.H{"name": name, "age": age})
	})

	r.Run(":8080")
}
