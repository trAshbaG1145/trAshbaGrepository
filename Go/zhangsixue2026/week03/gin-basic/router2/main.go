package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()
	api := r.Group("/api")
	{
		v1 := api.Group("/v1")
		{
			v1.GET("/user", func(c *gin.Context) {
				c.String(http.StatusOK, "API v1 - User")
			})
			v1.GET("/product", func(c *gin.Context) {
				c.String(http.StatusOK, "API v1 - Product")
			})
		}
		v2 := api.Group("/v2")
		{
			v2.GET("/user", func(c *gin.Context) {
				c.String(http.StatusOK, "API v2 - User")
			})
			v2.GET("/product", func(c *gin.Context) {
				c.String(http.StatusOK, "API v2 - Product")
			})
		}
	}
	r.Run(":8080")
}
