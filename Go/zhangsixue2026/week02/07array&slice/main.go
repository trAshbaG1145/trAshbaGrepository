package main

import "fmt"

func modifyArray(arr [3]int) {
	arr[0] = 100 // 修改的是副本，不会影响原数组
}

func modifySlice(slice []int) {
	slice[0] = 100 // 修改的是底层数组，会影响原切片
}

func main() {
	fmt.Println("Hello, World!")
	//1.数组
	arr2 := [3]int{1, 2, 3} // 声明并初始化
	arr3 := [3]int{1, 2, 3} // 声明并初始化

	fmt.Println(arr2 == arr3)

	fmt.Println(&arr2[0])
	fmt.Println(&arr3[0])

	//2.切片
	slice2 := []int{1, 2, 3} // 声明并初始化

	fmt.Println(&slice2[0])

	slice3 := slice2 //赋值操作，只复制切片头(使用copy复制时，内存分配会不同。)

	fmt.Println(&slice3[0])

	slice4 := append(slice3, 4)
	// 指针变了
	fmt.Println(&slice3[0], &slice4[0])

	fmt.Println("new")
	s := make([]int, 0, 2)
	s1 := append(s, 1)
	s2 := append(s, 2, 3)
	fmt.Println(&s1[0], &s2[0])

	// 3.函数参数是值传递
	// 数组示例
	arr := [3]int{1, 2, 3}
	modifyArray(arr)
	fmt.Println("数组:", arr) // 输出: [1 2 3]

	// 切片示例
	slice := []int{1, 2, 3}
	modifySlice(slice)
	fmt.Println("切片:", slice) // 输出: [100 2 3]

}
