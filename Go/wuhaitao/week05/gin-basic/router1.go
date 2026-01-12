package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()
	r.GET("/user/:name", func(c *gin.Context) {
		name := c.Param("name")
		c.String(http.StatusOK, "Hello<b>dd</b>, %s!", name)
	})

	r.GET("/usernumber/{id:[0-9]+}", func(c *gin.Context) {
		id := c.Param("id")
		c.String(http.StatusOK, "User ID: %s", id)
	})
	r.Run(":8080")
}
