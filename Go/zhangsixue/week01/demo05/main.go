package main

import (
	"fmt"
)

// switch使用
func switchfunc() {
	month := 2
	switch month {
	case 3, 4, 5:
		fmt.Println("春天")
	case 6, 7, 8:
		fmt.Println("夏天")
	case 9, 10, 11:
		fmt.Println("秋天")
	case 12, 1, 2:
		fmt.Println("冬天")
	default:
		fmt.Println("输入有误...")
	}
}

// switch穿透能力
func switchfunc2() {
	s := "hello"
	switch {
	case s == "hello":
		fmt.Println("hello")
		fallthrough
	case s != "world":
		fmt.Println("world")
	}
}

// for例子
func forfunc() {
	s := "abc中国"
	for i, n := 0, len(s); i < n; i++ { // 常见的 for 循环，支持初始化语句。
		println(s[i])
	}
}

// for range
func forangefunc() {
	s := "abc中国"
	for i, value := range s {
		println(i, value)
	}
}

// break continue例子

func breakExample() {
	for i := 0; i < 10; i++ {
		if i == 5 {
			break
		}
		fmt.Println("Break example:", i)
	}
}

func continueExample() {
	for i := 0; i < 10; i++ {
		if i%2 == 0 {
			continue
		}
		fmt.Println("Continue example:", i)
	}
}
func main() {
	// switchfunc2()
	// forangefunc()
	breakExample()
}
