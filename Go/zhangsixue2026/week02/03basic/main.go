package main

import (
	"fmt"
	"unsafe"
)

func init() {
	fmt.Println("I am init1")
}
func init() {
	fmt.Println("I am init2")
}

var s1 string
var s2 string = "hello"
var s3 = "world"
var s4, s5 string = "hello", "world"

const c1 = "c1"

var (
	s6 = "s6"
	s7 = "s7"
	c2 = "c2"
)

func getTwoValues() (string, int) {
	return "hello", 42
}
func main() {
	fmt.Println("Hello, World!")
	fmt.Println(s1)
	fmt.Println(s2)
	fmt.Println(s3)
	fmt.Println(s4)
	fmt.Println(s5)

	s8 := "s8"
	fmt.Println(s8)

	s9, i1 := getTwoValues()
	fmt.Println(s9, i1)

	var num int
	size := unsafe.Sizeof(num)
	fmt.Printf("int类型的存储大小为: %d 字节\n", size)

	var a byte = 65
	fmt.Printf("%c\n", a)

}
