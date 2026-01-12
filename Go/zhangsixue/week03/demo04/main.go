package main

import (
	"fmt"
	"time"
)

func mygo(name string) {
	for i := 0; i < 10; i++ {
		fmt.Printf("In goroutine %s\n", name)
		// 为了避免第一个协程执行过快，观察不到并发的效果，加个休眠
		time.Sleep(10 * time.Millisecond)
	}
}

// 由于 x=x+1 不是原子操作
// 所以应避免多个协程对x进行操作
// 使用容量为1的信道可以达到锁的效果
func increment(ch chan bool, x *int) {
	ch <- true
	*x = *x + 1
	<-ch
}
func main() {
	// demo01
	// go mygo("协程1号") // 第一个协程
	// go mygo("协程2号") // 第二个协程
	// time.Sleep(time.Second)

	// demo02
	// pipline := make(chan int)
	// // 往信道中发送数据
	// go func() {
	// 	pipline <- 200
	// }()

	// // 从信道中取出数据，并赋值给mydata
	// mydata, ok := <-pipline

	// fmt.Println(mydata, ok)

	// demo03
	// pipline := make(chan int)

	// go func() {
	// 	time.Sleep(time.Second * 3)
	// 	fmt.Println("准备发送数据: 100")
	// 	pipline <- 100
	// }()

	// go func() {
	// 	num := <-pipline
	// 	fmt.Printf("接收到的数据是: %d", num)
	// }()
	// // 主函数sleep，使得上面两个goroutine有机会执行
	// time.Sleep(time.Second * 5)

	//demo04
	// var x int
	// for i := 0; i < 1000; i++ {
	// 	go func(x *int) {
	// 		*x = *x + 1
	// 	}(&x)
	// }
	// fmt.Println("x 的值：", x)

	// demo05
	// 注意要设置容量为 1 的缓冲信道
	pipline := make(chan bool, 1)

	var x int
	for i := 0; i < 1000; i++ {
		go increment(pipline, &x)
	}

	// 确保所有的协程都已完成
	// 以后会介绍一种更合适的方法（Mutex），这里暂时使用sleep
	time.Sleep(time.Second)
	fmt.Println("x 的值：", x)

	//demo06

	// var num int
	// var mu sync.Mutex
	// var wg sync.WaitGroup
	// for i := 0; i < 1000; i++ {
	// 	wg.Add(1)
	// 	go func() {
	// 		defer wg.Done()
	// 		mu.Lock()
	// 		num++ // 复合操作，非线程安全
	// 		mu.Unlock()

	// 	}()
	// }
	// wg.Wait()
	// fmt.Println("最终值:", num)

}
