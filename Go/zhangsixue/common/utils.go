package common

import "fmt"

func init() {
	fmt.Println("this is func utils 1")
}
func init() {
	fmt.Println("this is func utils 2")
}

// PrintHello prints a hello message
func PrintHello() {
	fmt.Println("Hello from common utilities!")
	printSome()
}

func printSome() {
	fmt.Println("Hello, this is PrintSome function!")
}
