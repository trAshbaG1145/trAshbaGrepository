package main

import (
	"fmt"
	"time"
)

func hello() {
	fmt.Println("Hello Goroutine!")
}
func main() {
	go hello()
	fmt.Println("main goroutine done!")
	time.Sleep(time.Second)
}

// 在Go语言中，主goroutine（main goroutine）结束时，程序会立即退出，不会等待其他goroutine完成。
// 当我们在main函数中使用go hello()启动一个新的goroutine时，主goroutine会继续执行后面的代码。
// 如果没有time.Sleep(time.Second)，主goroutine可能会在hello() goroutine执行完之前就结束，导致我们看不到"Hello Goroutine!"的输出。
// time.Sleep(time.Second)的作用是让主goroutine暂停1秒钟，给hello() goroutine足够的时间来执行并输出结果。
// 在实际开发中，我们通常使用sync.WaitGroup或channel来同步goroutine，而不是使用time.Sleep。
