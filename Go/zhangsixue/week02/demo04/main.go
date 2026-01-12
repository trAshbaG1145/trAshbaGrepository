package main

import (
	"fmt"
	"os"
)

// 文件写入 [在nodejs中，路径会根据当前代码计算，因为执行的入口路径可变。]
func m1() {
	file, err := os.OpenFile("writefile.txt", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		fmt.Println("无法打开或创建文件:", err)
		return
	}
	defer file.Close()

	data := []byte("这是要写入文件的数据")
	n, err := file.Write(data)
	if err != nil {
		fmt.Println("写入文件时出错:", err)
		return
	}
	fmt.Printf("成功写入 %d 个字节\n", n)
}
func main() {
	fmt.Println("Hello, World!")
	m1()
}
