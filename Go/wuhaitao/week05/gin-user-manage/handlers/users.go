package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"user-manage/validators"

	"github.com/gin-gonic/gin"
)

type User struct {
	Name  string `form:"name"`
	Email string `form:"email" binding:"required,email"`
	Age   int    `form:"age"`
}

// UserService 用于处理用户相关操作的服务
type UserService struct {
	filePath string
}

// NewUserService 创建一个新的用户服务实例
func NewUserService(filePath string) *UserService {
	return &UserService{
		filePath: filePath,
	}
}

// WriteUsers 将用户数据写入文件
func (s *UserService) WriteUsers(users []User) error {
	jsonFile, err := os.Create(s.filePath)
	if err != nil {
		return err
	}
	defer jsonFile.Close()
	return json.NewEncoder(jsonFile).Encode(users)
}

// GetUsers 从文件获取用户数据
func (s *UserService) GetUsers() ([]User, error) {
	fmt.Println("GetUsers", s.filePath)
	jsonFile, err := os.Open(s.filePath)
	if err != nil {
		return nil, err
	}
	defer jsonFile.Close()

	byteValue, _ := io.ReadAll(jsonFile)
	var users []User
	json.Unmarshal(byteValue, &users)
	return users, nil
}

// IsEmailExists 检查邮箱是否已存在
func (s *UserService) IsEmailExists(email string) (bool, error) {
	users, err := s.GetUsers()
	if err != nil {
		return false, err
	}

	for _, user := range users {
		if user.Email == email {
			return true, nil
		}
	}
	return false, nil
}

// AddUser 添加新用户
func (s *UserService) AddUser(user User) error {
	users, err := s.GetUsers()
	if err != nil {
		return err
	}

	users = append(users, user)
	return s.WriteUsers(users)
}

// UpdateUser 更新用户信息
func (s *UserService) UpdateUser(user User) (bool, error) {
	users, err := s.GetUsers()
	if err != nil {
		return false, err
	}

	found := false
	for i, u := range users {
		if u.Email == user.Email {
			users[i] = user
			found = true
			break
		}
	}

	if !found {
		return false, nil
	}

	return true, s.WriteUsers(users)
}

// DeleteUser 根据邮箱删除用户
func (s *UserService) DeleteUser(email string) (bool, error) {
	users, err := s.GetUsers()
	if err != nil {
		return false, err
	}

	found := false
	for i, u := range users {
		if u.Email == email {
			users = append(users[:i], users[i+1:]...)
			found = true
			break
		}
	}

	if !found {
		return false, nil
	}

	return true, s.WriteUsers(users)
}

// 创建全局用户服务实例
var userService = NewUserService("user.json")

// GetUsersHandler 获取所有用户的处理函数
func GetUsersHandler(c *gin.Context) {
	users, err := userService.GetUsers()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": -1, "error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, users)
}

// CreateUserHandler 创建用户的处理函数
func CreateUserHandler(c *gin.Context) {
	var user User
	if err := c.ShouldBind(&user); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": -1, "error": validators.TranslateError(err)})
		return
	}

	exists, err := userService.IsEmailExists(user.Email)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": -1, "error": err.Error()})
		return
	}

	if exists {
		c.JSON(http.StatusOK, gin.H{"code": -1, "message": "该邮箱已被注册"})
		return
	}

	err = userService.AddUser(user)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": -1, "error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"code": 0, "message": "用户创建成功"})
}

// UpdateUserHandler 更新用户的处理函数
func UpdateUserHandler(c *gin.Context) {
	var user User
	if err := c.ShouldBind(&user); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": -1, "error": validators.TranslateError(err)})
		return
	}

	updated, err := userService.UpdateUser(user)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": -1, "error": err.Error()})
		return
	}

	if !updated {
		c.JSON(http.StatusOK, gin.H{"code": -1, "message": "未找到该用户"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"code": 0, "message": "更新用户成功"})
}

// DeleteUserHandler 删除用户的处理函数
func DeleteUserHandler(c *gin.Context) {
	email := c.Param("email")

	deleted, err := userService.DeleteUser(email)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": -1, "error": err.Error()})
		return
	}

	if !deleted {
		c.JSON(http.StatusOK, gin.H{"code": -1, "message": "未找到该用户"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"code": 0, "message": "删除用户成功"})
}
