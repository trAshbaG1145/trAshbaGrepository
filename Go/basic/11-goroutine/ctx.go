package main

import (
	"context"
	"fmt"
	"time"
)

func doSomething(ctx context.Context) {
    // 每隔 1 秒检查是否超时
    for {
        select {
        case <-ctx.Done():
            fmt.Println("操作已被取消或超时")
            return
        default:
            fmt.Println("正在执行操作...")
            time.Sleep(1 * time.Second)
        }
    }
}

func main() {
    // 设置 5 秒的截止时间
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    go doSomething(ctx)

    time.Sleep(10 * time.Second)
}