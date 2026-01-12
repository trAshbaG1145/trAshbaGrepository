package main

import (
	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()
	r.LoadHTMLFiles("user.tmpl")

	r.GET("/template", func(c *gin.Context) {
		users := []struct {
			Name string
			Age  int
		}{
			{"张三", 25},
			{"李四", 30},
			{"王五", 28},
		}

		c.HTML(200, "user.tmpl", gin.H{
			"title":      "用户列表",
			"users":      users,
			"showAdmins": true,
			"admins":     []string{"admin1", "admin2"},
		})
	})
	r.Run(":8080")
}
