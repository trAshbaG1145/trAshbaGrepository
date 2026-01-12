package main

import (
	"fmt"
	filehandles "gin-basic/file/file-handles"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()
	r.POST("/upload", filehandles.Upload)
	r.POST("/uploads", filehandles.Uploads)

	fmt.Println("服务启动成功：http://localhost:8080")
	r.Run(":8080")
}
