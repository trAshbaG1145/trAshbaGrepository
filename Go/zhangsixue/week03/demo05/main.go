package main

import (
	"fmt"
	_ "go-basics/zhangsixue/common"
)

func init() {
	fmt.Println("this is func main")
}
func main() {
	// common.PrintHello()
	// common.printSome() //小写私有，其他包不可调用
	fmt.Println("Hello, World!")
}
