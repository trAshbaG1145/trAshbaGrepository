package main

import (
	"fmt"
	"time"
)

// select示例：演示如何使用select语句来处理channel的写入操作
// 本例子展示了如何：
// 1. 使用带缓冲的channel
// 2. 使用select进行非阻塞写入
// 3. 检测channel是否已满
func main() {
	// 创建一个带缓冲的channel，缓冲区大小为10
	// 这意味着可以存储10个string类型的数据而不阻塞
	output1 := make(chan string, 10)

	// 启动一个goroutine执行write函数
	// 该goroutine会持续向channel写入数据
	go write(output1)

	// 主goroutine从channel中读取数据
	// 使用for range循环，会一直从channel读取数据直到channel被关闭
	for s := range output1 {
		fmt.Println("res:", s)
		// 每读取一个数据就暂停1秒
		// 这样可以模拟一个慢速的消费者
		time.Sleep(time.Second)
	}
}

// write 函数持续向channel写入数据
// ch: 写入目标channel
func write(ch chan string) {
	// 无限循环，持续尝试写入数据
	for {
		// select语句用于处理多个channel操作
		// 这里我们用它来实现非阻塞的写入操作
		select {
		// 尝试向channel写入"hello"
		case ch <- "hello":
			fmt.Println("write hello")
		// default分支在所有case都阻塞时执行
		// 这里用来处理channel已满的情况
		default:
			fmt.Println("channel full")
		}
		// 每次循环暂停500毫秒
		// 这样可以控制写入速度，方便观察现象
		time.Sleep(time.Millisecond * 500)
	}
}
