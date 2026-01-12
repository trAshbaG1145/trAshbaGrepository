package main

import (
	"fmt"
)

func main() {
	// 1. 切片的基本概念
	fmt.Println("===== 切片的基本概念 =====")
	/*
	   切片(slice)是Go语言中的一种动态数组，它是对数组的一种抽象和封装
	   切片本身不存储任何数据，它只是底层数组的一个引用
	   切片有三个属性：指针、长度和容量
	   - 指针：指向切片第一个元素对应的底层数组元素的地址
	   - 长度：切片中元素的数量，通过len()函数获取
	   - 容量：从切片的起始元素到底层数组末尾的元素数量，通过cap()函数获取
	*/

	// 2. 创建切片的几种方式
	fmt.Println("\n===== 创建切片的几种方式 =====")

	// 2.1 使用make函数创建切片
	// make([]T, length, capacity)
	slice1 := make([]int, 5, 10) // 长度为5，容量为10的int切片
	fmt.Printf("slice1: %v, 长度: %d, 容量: %d\n", slice1, len(slice1), cap(slice1))

	// 2.2 使用字面量创建切片
	slice2 := []int{1, 2, 3, 4, 5}
	fmt.Printf("slice2: %v, 长度: %d, 容量: %d\n", slice2, len(slice2), cap(slice2))

	// 2.3 从数组创建切片
	array := [5]int{10, 20, 30, 40, 50}
	slice3 := array[1:4] // 从索引1到索引3（不包括4）
	fmt.Printf("原数组: %v\n", array)
	fmt.Printf("slice3: %v, 长度: %d, 容量: %d\n", slice3, len(slice3), cap(slice3))

	// 3. 切片的基本操作
	fmt.Println("\n===== 切片的基本操作 =====")

	// 3.1 访问和修改元素
	fmt.Println("原始slice2:", slice2)
	slice2[2] = 100 // 修改第三个元素
	fmt.Println("修改后slice2:", slice2)

	// 3.2 切片的切片
	subSlice := slice2[1:3]
	fmt.Printf("subSlice: %v, 长度: %d, 容量: %d\n", subSlice, len(subSlice), cap(subSlice))

	// 3.3 追加元素 - append函数
	slice4 := []int{1, 2, 3}
	fmt.Printf("原始slice4: %v, 长度: %d, 容量: %d\n", slice4, len(slice4), cap(slice4))

	slice4 = append(slice4, 4, 5)
	fmt.Printf("append后slice4: %v, 长度: %d, 容量: %d\n", slice4, len(slice4), cap(slice4))

	// 3.4 合并切片
	slice5 := []int{6, 7, 8}
	slice4 = append(slice4, slice5...)
	fmt.Printf("合并后slice4: %v, 长度: %d, 容量: %d\n", slice4, len(slice4), cap(slice4))

	// 4. 切片的扩容机制演示
	fmt.Println("\n===== 切片的扩容机制 =====")
	growingSlice := make([]int, 0)
	fmt.Printf("初始状态: %v, 长度: %d, 容量: %d\n", growingSlice, len(growingSlice), cap(growingSlice))

	// 追加元素观察容量变化
	for i := 0; i < 10; i++ {
		growingSlice = append(growingSlice, i)
		fmt.Printf("添加元素 %d 后: 长度: %d, 容量: %d\n", i, len(growingSlice), cap(growingSlice))
	}

	// 5. 切片与数组的区别
	fmt.Println("\n===== 切片与数组的区别 =====")
	/*
	   1. 数组是固定长度的，切片是动态长度的
	   2. 数组是值类型（复制时会复制整个数组），切片是引用类型（复制时只复制引用）
	   3. 数组作为函数参数时会复制整个数组，切片作为函数参数时只会复制引用
	*/

	// 演示切片是引用类型
	originalSlice := []int{1, 2, 3}
	copySlice := originalSlice

	fmt.Println("修改前:")
	fmt.Println("originalSlice:", originalSlice)
	fmt.Println("copySlice:", copySlice)

	copySlice[0] = 100

	fmt.Println("修改后:")
	fmt.Println("originalSlice:", originalSlice)
	fmt.Println("copySlice:", copySlice)

	// 6. nil切片和空切片
	fmt.Println("\n===== nil切片和空切片 =====")
	var nilSlice []int    // nil切片
	emptySlice := []int{} // 空切片

	fmt.Println("nilSlice == nil:", nilSlice == nil)
	fmt.Println("emptySlice == nil:", emptySlice == nil)
	fmt.Printf("nilSlice: %v, 长度: %d, 容量: %d\n", nilSlice, len(nilSlice), cap(nilSlice))
	fmt.Printf("emptySlice: %v, 长度: %d, 容量: %d\n", emptySlice, len(emptySlice), cap(emptySlice))
}
