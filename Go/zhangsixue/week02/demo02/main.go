package main

import "fmt"

func m1() {
	scores := map[string]int{"english": 80, "chinese": 85}
	if math, ok := scores["math"]; ok {
		fmt.Printf("math 的值是: %d", math)
	} else {
		fmt.Println("math 不存在")
	}
}

func m2() {
	scores := map[string]int{"english": 80, "chinese": 85}

	for subject, score := range scores {
		fmt.Printf("key: %s, value: %d\n", subject, score)
	}
}

func m3() {
	scores := map[string]int{"english": 80, "chinese": 85}

	for subject := range scores {
		fmt.Printf("key: %s\n", subject)
	}
}

func m4() {
	scores := map[string]int{"english": 80, "chinese": 85}

	for _, score := range scores {
		fmt.Printf("value: %d\n", score)
	}
}
func main() {
	fmt.Println("Hello, World!")

	m1()
	m2()
	m3()
	m4()

}
