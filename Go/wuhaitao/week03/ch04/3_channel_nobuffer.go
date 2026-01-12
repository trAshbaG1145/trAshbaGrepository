// 无缓冲通道
package main

import (
	"fmt"
	"time"
)

func recv(c chan int) {
	time.Sleep(time.Second * 2)

	fmt.Println("开始接收")

	ret := <-c
	fmt.Println("接收成功", ret)

}
func main() {
	ch := make(chan int)

	go recv(ch) // 启用goroutine从通道接收值

	fmt.Println("开始发送10")

	ch <- 10

	fmt.Println("发送成功10")

	go recv(ch)

	ch <- 11

	fmt.Println("发送成功11")

	time.Sleep(time.Second * 2)

}
