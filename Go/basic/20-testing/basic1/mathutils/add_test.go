package mathutils

import "testing"

// TestAdd 是一个测试Add函数的单元测试
func TestAdd(t *testing.T) {
	result := Add(2, 3)
	expected := 50

	if result != expected {
		t.Errorf("Add(2, 3) = %d; want %d", result, expected)
	}
}
