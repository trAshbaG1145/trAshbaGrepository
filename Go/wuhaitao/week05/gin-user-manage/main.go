package main

import (
	"fmt"
	"log"
	"os"
	"user-manage/handlers"
	"user-manage/validators"

	"github.com/gin-gonic/gin"
)

func main() {
	// 初始化验证器
	validators.InitValidator()

	router := gin.Default()

	//检查user.json文件是否存在
	if _, err := os.Stat("user.json"); os.IsNotExist(err) {
		log.Println("user.json文件不存在，创建文件")
		os.Create("user.json")
	}

	router.GET("/users", handlers.GetUsersHandler)
	router.POST("/users", handlers.CreateUserHandler)
	router.PUT("/users", handlers.UpdateUserHandler)
	router.DELETE("/users/:email", handlers.DeleteUserHandler)

	fmt.Println("user.json文件存在")

	log.Println("服务器启动，监听端口 :8080")
	router.Run(":8080")
}
