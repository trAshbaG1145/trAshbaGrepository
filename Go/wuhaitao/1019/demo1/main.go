package main

import (
	_ "demo1/calc"
	"fmt"

	"github.com/google/uuid"
)

func main() {

	uuid := uuid.New()
	fmt.Println(uuid)

	// fmt.Println(calc.Add(1, 2))
}
