package main

import (
	"fmt"
)

func main() {
	// 1. 数组的声明与初始化
	fmt.Println("---- 数组的声明与初始化 ----")

	// 声明固定长度的数组
	var arr1 [5]int
	fmt.Printf("arr1: %v, 类型: %T\n", arr1, arr1)

	// 声明并初始化
	arr2 := [5]int{1, 2, 3, 4, 5}
	fmt.Printf("arr2: %v\n", arr2)

	// 使用...让编译器自动计算长度
	arr3 := [...]int{10, 20, 30, 40}
	fmt.Printf("arr3: %v, 长度: %d\n", arr3, len(arr3))

	// 指定索引的初始化
	arr4 := [5]int{0: 100, 2: 300, 4: 500}
	fmt.Printf("arr4: %v\n", arr4)

	// 2. 数组的访问与修改
	fmt.Println("\n---- 数组的访问与修改 ----")

	// 访问数组元素
	fmt.Printf("arr2[0]: %d, arr2[4]: %d\n", arr2[0], arr2[4])

	// 修改数组元素
	arr2[0] = 100
	arr2[4] = 500
	fmt.Printf("修改后的arr2: %v\n", arr2)

	// 3. 数组的遍历
	fmt.Println("\n---- 数组的遍历 ----")

	// 使用for循环遍历
	fmt.Println("使用for循环:")
	for i := 0; i < len(arr3); i++ {
		fmt.Printf("arr3[%d] = %d\n", i, arr3[i])
	}

	// 使用for-range遍历
	fmt.Println("\n使用for-range循环:")
	for index, value := range arr3 {
		fmt.Printf("arr3[%d] = %d\n", index, value)
	}

	// 仅获取值
	fmt.Println("\n仅获取数组值:")
	for _, value := range arr3 {
		fmt.Printf("%d ", value)
	}
	fmt.Println()

	// 4. 多维数组
	fmt.Println("\n---- 多维数组 ----")

	// 声明并初始化二维数组
	matrix := [3][4]int{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}
	fmt.Println("二维数组:")
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			fmt.Printf("%3d ", matrix[i][j])
		}
		fmt.Println()
	}

	// 5. 数组作为函数参数
	fmt.Println("\n---- 数组作为函数参数 ----")

	// 值传递 - 原数组不会被修改
	testArray := [5]int{1, 2, 3, 4, 5}
	fmt.Printf("调用函数前: %v\n", testArray)
	modifyArray(testArray)
	fmt.Printf("调用函数后: %v（注意：没有改变，因为是值传递）\n", testArray)

	// 使用指针 - 原数组会被修改
	fmt.Printf("调用指针函数前: %v\n", testArray)
	modifyArrayByPointer(&testArray)
	fmt.Printf("调用指针函数后: %v（值已改变，因为传递了指针）\n", testArray)
}

// 值传递函数 - 不会修改原数组
func modifyArray(arr [5]int) {
	arr[0] = 100
	fmt.Printf("在函数内: %v\n", arr)
}

// 指针传递函数 - 会修改原数组
func modifyArrayByPointer(arr *[5]int) {
	arr[0] = 100
	fmt.Printf("在指针函数内: %v\n", *arr)
}
