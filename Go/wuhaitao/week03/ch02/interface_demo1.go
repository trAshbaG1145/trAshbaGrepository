package main

import (
	"fmt"
	"math"
)

// Shape 是一个接口，定义了计算面积的方法
type Shape interface {
	Area() float64
	Name() string
}

// Rectangle 矩形结构体
type Rectangle struct {
	Width  float64
	Height float64
}

// 实现Shape接口的Area方法
func (r Rectangle) Area() float64 {
	return r.Width * r.Height
}

// 实现Shape接口的Name方法
func (r Rectangle) Name() string {
	return "矩形"
}

// Circle 圆形结构体
type Circle struct {
	Radius float64
}

// 实现Shape接口的Area方法
func (c Circle) Area() float64 {
	return math.Pi * c.Radius * c.Radius
}

// 实现Shape接口的Name方法
func (c Circle) Name() string {
	return "圆形"
}

// Triangle 三角形结构体
type Triangle struct {
	Base   float64
	Height float64
}

// 实现Shape接口的Area方法
func (t Triangle) Area() float64 {
	return 0.5 * t.Base * t.Height
}

// 实现Shape接口的Name方法
func (t Triangle) Name() string {
	return "三角形"
}

// 打印形状信息的函数，接收Shape接口类型
func printShapeInfo(s Shape) {
	fmt.Printf("%s的面积是: %.2f\n", s.Name(), s.Area())
}

func main() {
	// 创建不同的形状实例
	rect := Rectangle{Width: 5, Height: 4}
	circle := Circle{Radius: 3}
	triangle := Triangle{Base: 6, Height: 8}

	// 创建Shape切片，存储不同的形状
	shapes := []Shape{rect, circle, triangle}

	// 使用for循环遍历所有形状并计算面积
	fmt.Println("计算所有形状的面积:")
	for _, shape := range shapes {
		printShapeInfo(shape)
	}

	// 直接使用接口变量
	var s Shape
	s = Rectangle{Width: 10, Height: 2}
	fmt.Printf("\n另一个%s的面积是: %.2f\n", s.Name(), s.Area())

	s = Circle{Radius: 5}
	fmt.Printf("另一个%s的面积是: %.2f\n", s.Name(), s.Area())
}
