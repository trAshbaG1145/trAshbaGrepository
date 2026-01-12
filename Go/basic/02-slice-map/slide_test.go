package main

import (
	"fmt"
	"testing"
)


func changeSlice(s []int) {
	s[0] = 100
}

func TestChangeSlice(t *testing.T) {
	s := []int{1, 2, 3}
	changeSlice(s)
	fmt.Println(s)
}

