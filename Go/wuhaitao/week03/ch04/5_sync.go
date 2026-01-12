package main

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"
)

var myArray = []string{"1,2", "3,4", "5,6", "7,8", "9,10"}

func main() {
	fmt.Println("Hello, World!")

	//这个channel，用len(myArray)最好，因为不会阻塞，也不会溢出
	ch := make(chan string, len(myArray))
	var wg sync.WaitGroup

	wg.Add(2)
	go sender(ch, &wg)
	go calculate(ch, &wg)

	wg.Wait()
}

func sender(ch chan string, wg *sync.WaitGroup) {
	defer wg.Done()
	for _, v := range myArray {
		// 注意：如果ch没有被读取，会阻塞，所以后面的close不会马上执行
		ch <- v
	}

	close(ch)
}

func calculate(ch chan string, wg *sync.WaitGroup) {

	for v := range ch {

		// 模拟计算时间
		time.Sleep(time.Second * 1)
		fmt.Println("calculate", v)

		numArray := strings.Split(v, ",")

		num1, _ := strconv.Atoi(numArray[0])
		num2, _ := strconv.Atoi(numArray[1])

		fmt.Printf("%d + %d = %d\n", num1, num2, num1+num2)
	}
	wg.Done()
}
