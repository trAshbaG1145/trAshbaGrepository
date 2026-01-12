package mathutils

import "errors"

// Divide 执行除法运算，并处理除以0的错误
func Divide(a, b int) (int, error) {
	if b == 0 {
		return 0, errors.New("division by zero")
	}
	return a / b, nil
}
