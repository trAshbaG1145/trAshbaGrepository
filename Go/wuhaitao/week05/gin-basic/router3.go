package main

import (
	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()
	r.GET("/search", func(c *gin.Context) {
		query := c.Query("q")
		page := c.DefaultQuery("page", "1")
		c.JSON(200, gin.H{
			"query": query,
			"page":  page,
		})
	})
	r.Run(":8080")
}
