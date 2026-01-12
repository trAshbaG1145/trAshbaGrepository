package main

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

func sendRequestWithTimeout() {
    // 创建一个带有 5 秒超时的上下文
    ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
    defer cancel()

    // 发送 HTTP 请求
    req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://www.facebook.com/", nil)
    if err!= nil {
        fmt.Println("创建请求时出错:", err)
        return
    }

    client := &http.Client{}
    resp, err := client.Do(req)
    if err!= nil {
        // 如果是因为超时导致的错误
        if ctx.Err() == context.DeadlineExceeded {
            fmt.Println("请求超时")
        } else {
            fmt.Println("请求发送时出错:", err)
        }
        return
    }
	defer resp.Body.Close()

    // 处理响应
    fmt.Println("收到响应")
}

func main() {
    sendRequestWithTimeout()
}