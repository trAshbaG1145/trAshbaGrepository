package main

import (
	"fmt"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()

	fmt.Println("服务启动成功：http://localhost:8080")
	r.Run(":8080")
}
