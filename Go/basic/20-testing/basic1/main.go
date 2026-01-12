package main

import (
	"basic1/mathutils"
	"fmt"
	"log"
)

func main() {
	// 调用加法函数
	sum := mathutils.Add(2, 3)
	fmt.Printf("2 + 3 = %d\n", sum)

	// 调用除法函数
	result, err := mathutils.Divide(10, 2)
	if err != nil {
		log.Fatalf("除法错误: %v\n", err)
	} else {
		fmt.Printf("10 ÷ 2 = %d\n", result)
	}

	// 尝试除以零
	_, err = mathutils.Divide(10, 0)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
	}
}
