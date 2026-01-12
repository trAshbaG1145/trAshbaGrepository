package main

import "fmt"

func double(n *int) {
	*n = *n * 2 // 修改指针指向的值
}
func double2(n int) {
	n = n * 2 // 修改指针指向的值
}

func modify(arr *[3]int) {
	(*arr)[0] = 90
}

func modify4(sls []int) {
	sls[0] = 90
}

// 声明一个 Profile 的结构体
type Profile struct {
	name   string
	age    int
	gender string
	mother *Profile // 指针
	father *Profile // 指针
}

// 重点在于这个星号: *
func (person *Profile) increase_age() {
	person.age += 1
}

// 指针的应用3，4,5
func main() {
	fmt.Println("Hello, World!")

	var a int = 10  // 直接存数据：把数字 10 放进一个盒子
	var p *int = &a // 指针：生成一张快递单，写上盒子的地址

	// demo1
	fmt.Println(a) // 10
	fmt.Println(p)

	// demo2
	double2(a)     //
	fmt.Println(a) //

	double(&a)     //
	fmt.Println(a) //

	// demo 3
	a3 := [3]int{89, 90, 91}
	modify(&a3)
	fmt.Println(a3)

	// demo4
	a4 := [3]int{89, 90, 91}
	modify4(a4[:])
	fmt.Println(a4)

	// demo5
	myself := Profile{name: "小明", age: 24, gender: "male"}
	fmt.Printf("当前年龄：%d\n", myself.age)
	myself.increase_age()
	fmt.Printf("当前年龄：%d", myself.age)

}
