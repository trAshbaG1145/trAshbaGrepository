/*
**
这个示例展示了 select 的几个重要特性：
创建了两个通道 ch1 和 ch2，分别由不同的 goroutine 向其发送数据
ch1 每2秒发送一次数据
ch2 每3秒发送一次数据
主 goroutine 使用 select 语句同时监听两个通道
包含了超时处理机制（使用 time.After）
运行这个程序，您将看到：
程序会交替接收来自 ch1 和 ch2 的消息
如果两个通道都没有数据且等待超过4秒，将触发超时处理
select 会随机选择一个已经就绪的通道进行处理
这个示例很好地展示了 select 在处理多个通道时的用法，这在并发编程中非常有用。
*/
package main

import (
	"fmt"
	"time"
)

func main() {
	// 创建两个通道
	ch1 := make(chan string)
	ch2 := make(chan string)

	// 启动第一个goroutine，向ch1发送数据
	go func() {
		for i := 0; i < 5; i++ {
			time.Sleep(time.Second * 2)
			ch1 <- fmt.Sprintf("来自ch1的消息 %d", i)
		}
	}()

	// 启动第二个goroutine，向ch2发送数据
	go func() {
		for i := 0; i < 5; i++ {
			time.Sleep(time.Second * 3)
			ch2 <- fmt.Sprintf("来自ch2的消息 %d", i)
		}
	}()

	// 使用select监听两个通道
	for i := 0; i < 10; i++ {
		select {
		case msg1 := <-ch1:
			fmt.Println("接收到:", msg1)
		case msg2 := <-ch2:
			fmt.Println("接收到:", msg2)
		case <-time.After(time.Second * 4):
			fmt.Println("超时等待...")
		}

		// time.Sleep(time.Second * 4)
	}
}

/**
进阶问题：time.After(time.Second * 4):这个语法是什么意思，一般用于什么场景
*/
