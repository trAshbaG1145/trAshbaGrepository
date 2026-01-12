package main

import (
	"fmt"
)

func main() {
	ch := make(chan int)
	go func() {
		for i := 0; i < 5; i++ {
			ch <- i
		}
		close(ch) // 必须关闭以退出 range 循环
	}()
	for v := range ch {
		fmt.Println(v)
	}
}
