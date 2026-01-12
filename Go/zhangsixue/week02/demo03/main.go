package main

import (
	"encoding/json"
	"fmt"
)

type Person struct {
	Name string
	Age  int
	City string
}

// 使用指针接收器定义一个方法
func (p *Person) IncreaseAge() {
	p.Age++
}

func m1() {
	p := Person{
		Name: "Charlie",
		Age:  28,
		City: "Los Angeles",
	}
	fmt.Println("修改前的年龄:", p.Age)
	p.IncreaseAge() // p没有带&
	fmt.Println("修改后的年龄:", p.Age)
}

/*

{
  "name": "John Doe",
  "age": 30,
  "isStudent": false,
  "hobbies": ["reading", "writing", "coding"],
  "address": {
    "street": "123 Main St",
    "city": "Anytown",
    "state": "CA",
    "zip": "12345"
  }
}
*/

func m2() {
	type Person struct {
		Name string
		Age  int
		City string
	}
	jsonStr := `{"Name":"Eve","Age":22,"City":"Seattle张"}`

	var p Person
	// 将JSON字符串反序列化为Person结构体
	err := json.Unmarshal([]byte(jsonStr), &p)
	if err != nil {
		fmt.Println("反序列化错误:", err)
		return
	}

	fmt.Printf("反序列化后的Person: %+v\n", p)

}

func m3() {
	type Person struct {
		Name string
		Age  int
		City string
	}
	p := Person{
		Name: "David",
		Age:  35,
		City: "Chicago",
	}

	// 将Person结构体序列化为JSON格式的字节切片
	data, err := json.MarshalIndent(p, "", "")
	if err != nil {
		fmt.Println("序列化错误:", err)
		return
	}

	fmt.Println("JSON字符串:", string(data))
}

func m4() {
	type student struct {
		id   int
		name string
		age  int
	}

	ce := make(map[int]student)
	ce[1] = student{1, "xiaolizi", 22}
	ce[2] = student{2, "wang", 23}
	fmt.Println(ce)
	delete(ce, 2)
	fmt.Println(ce)
}
func main() {
	// m1()
	// m2()
	// m3()
	// m4()
}
