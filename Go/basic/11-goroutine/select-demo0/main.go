/**
这个例子展示了 select 的基本用法：
1. 创建了两个通道 ch1 和 ch2
启动两个 goroutine 分别向这些通道发送消息
使用 select 语句等待接收消息
4. 包含了一个超时控制的 case
由于 ch2 的消息会在1秒后发送，而 ch1 的消息在2秒后发送，所以这个程序会打印"来自通道2的消息"，因为它是最先准备好的 case。
这个例子很好地展示了 select 的非阻塞特性：它会选择第一个准备好的通道进行操作。如果所有通道都没有准备好，并且超过了3秒，就会执行超时 case。
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

	// 启动一个goroutine，向ch1发送数据
	go func() {
		time.Sleep(2 * time.Second) // 延迟2秒
		ch1 <- "来自通道1的消息"
	}()

	// 启动另一个goroutine，向ch2发送数据
	go func() {
		time.Sleep(1 * time.Second) // 延迟1秒
		ch2 <- "来自通道2的消息"
	}()

	// select 语句会等待其中任意一个case可以执行时就执行
	select {
	case msg1 := <-ch1: // 从ch1接收数据
		fmt.Println(msg1)
	case msg2 := <-ch2: // 从ch2接收数据
		fmt.Println(msg2)
	case <-time.After(3 * time.Second): // 超时控制，3秒后如果没有收到消息就执行
		fmt.Println("超时了！")
	}
}
