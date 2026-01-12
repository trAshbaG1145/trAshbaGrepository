package calc

import "fmt"

func Add(a, b int) int {
	return a + b
}

func Sub(a, b int) int {
	return a - b
}

func Mul(a, b int) int {
	return a * b
}

func Div(a, b int) int {
	return a / b
}

func init() {
	fmt.Println("calc init")
}

func init() {
	fmt.Println("calc init2")
}
