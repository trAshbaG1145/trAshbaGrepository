package filehandles

import "github.com/gin-gonic/gin"

func Upload(c *gin.Context) {
	user := c.PostForm("user")
	file, _ := c.FormFile("file")
	c.SaveUploadedFile(file, "uploads/"+user+"_"+file.Filename)
	c.JSON(200, gin.H{
		"message": "上传成功",
	})
}

func Uploads(c *gin.Context) {
	user := c.PostForm("user")
	form, _ := c.MultipartForm()
	files := form.File["files"]
	for _, file := range files {
		c.SaveUploadedFile(file, "uploads/"+user+"_"+file.Filename)
	}
	c.JSON(200, gin.H{
		"message": "上传成功",
	})
}
