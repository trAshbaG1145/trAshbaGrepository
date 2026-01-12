package goroutine01

import (
	"fmt"
	"time"
)

func task() {
	fmt.Println("Hello from Goroutine")
	time.Sleep(2 * time.Second)
	fmt.Println("Goroutine completed")
}

func main() {
	go task()
	fmt.Println("Main function")
	time.Sleep(3 * time.Second)
}
