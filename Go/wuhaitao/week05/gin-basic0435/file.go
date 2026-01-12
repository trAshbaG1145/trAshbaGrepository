package main

import (
	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()

	r.POST("/upload", func(c *gin.Context) {
		// 单文件
		file, _ := c.FormFile("file")
		name := c.PostForm("name")
		c.SaveUploadedFile(file, "uploads/"+file.Filename)

		c.JSON(200, gin.H{
			"message": "上传成功",
			"name":    name,
		})
	})

	r.Run(":8080")
}
