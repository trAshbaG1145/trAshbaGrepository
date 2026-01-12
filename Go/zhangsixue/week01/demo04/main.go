package main

import (
	"fmt"
)

func funcDemo1(x, y int) int {
	// 类型相同的相邻参数，参数类型可合并。 多返回值必须用括号。
	n := x + y
	return n
}

func funcDemo2(a int, args ...int) int {

	sum := 0
	for _, arg := range args {
		fmt.Println(arg)
		sum += arg
	}
	return a + sum
}

func funcDemo3(a, b int) (sum int, avg int) {
	sum = a + b
	avg = (a + b) / 2
	return
}

func funcDemo4() func() int {
	i := 0
	b := func() int {
		i++
		fmt.Println(i)
		return i
	}
	return b
}

func myfunc() {
	fmt.Println("B")
}

func deferDemo() {
	var whatever [5]struct{}
	for i := range whatever {
		defer func() { fmt.Println(i) }()
	}
}
func main() {
	// defer myfunc()
	// funcDemo1(1, 2)
	// fmt.Println(funcDemo2(1, 2, 3, 4, 5, 6))
	// fmt.Println(funcDemo3(11, 22))

	// getSqrt := func(a float64) float64 {
	// 	return math.Sqrt(a)
	// }
	// fmt.Println(getSqrt(4))

	// fc4 := funcDemo4()
	// fmt.Println(fc4())

	deferDemo()

}
