package main

import (
	"fmt"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func getUserInfo(c *gin.Context) {
	cookie_uuid, err := c.Cookie("uuid")

	fmt.Println("getUserInfo", err)
	if err != nil {
		//302 redirect to login page
		c.Redirect(302, "/login")
		return
	}
	c.Set("cookie_uuid", cookie_uuid)
	c.Next()

}

func renderIndex(c *gin.Context) {
	cookie_uuid := c.GetString("cookie_uuid")
	c.String(200, "Hello, you are logged in, your uuid is "+cookie_uuid)
}

func renderUser(c *gin.Context) {
	c.String(200, "User Page, you are logged in")
}

func login(c *gin.Context) {

	//mock login,set cookie
	c.SetCookie("uuid", uuid.New().String(), 3600, "/", "localhost", false, true)
	c.Redirect(302, "/")
}

func logout(c *gin.Context) {
	c.SetCookie("uuid", "", -1, "/", "localhost", false, true)
	c.Redirect(302, "/")
}

func main() {
	r := gin.Default()

	r.GET("/", getUserInfo, renderIndex)
	r.GET("/user", getUserInfo, renderUser)
	r.GET("/login", login)
	r.GET("/logout", logout)

	r.Run()
}
