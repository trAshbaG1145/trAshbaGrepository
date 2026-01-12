package main

import "fmt"

func main() {

	myBytes := []byte("武汉abc123")
	printBytes(myBytes)
}

func printBytes(b []byte) {
	//10进制
	fmt.Println("10进制视图")
	for _, v := range b {
		fmt.Printf("%d ", v)
	}

	fmt.Println("\n16进制视图")
	for _, v := range b {
		fmt.Printf("%x ", v)
	}
}
