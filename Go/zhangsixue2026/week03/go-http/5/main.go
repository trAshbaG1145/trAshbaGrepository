package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
)

func main() {
	// 代理服务器地址
	proxyURL, err := url.Parse("http://127.0.0.1:8899")
	if err != nil {
		fmt.Println("解析代理服务器地址失败:", err)
		return
	}
	// 创建一个Transport对象，并设置代理
	transport := &http.Transport{
		Proxy: http.ProxyURL(proxyURL),
	}
	client := &http.Client{
		Transport: transport,
	}
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
	resp, err := client.Post("https://jsonplaceholder.typicode.com/posts",
		"application/json",
		bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Println("请求失败:", err)
		return
	}
	defer resp.Body.Close()

	fmt.Printf("状态码: %d\n", resp.StatusCode)
}
