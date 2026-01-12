package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

func main() {
	// 准备要发送的数据
	data := map[string]string{
		"name": "小明",
		"age":  "18",
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		fmt.Println("JSON编码失败:", err)
		return
	}

	// 发送POST请求
	resp, err := http.Post("https://jsonplaceholder.typicode.com/posts",
		"application/json",
		bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Println("请求失败:", err)
		return
	}
	defer resp.Body.Close()

	fmt.Printf("状态码: %d\n", resp.StatusCode)
}
