package main

import (
	"fmt"
	"unicode/utf8"
)

func main() {
	// rune 是 Go 的内置类型，实际是 int32 的别名
	// 用于表示 Unicode 码点（Code Point）

	// 1. 基本的 rune 定义和使用
	var r rune = '武'
	fmt.Printf("字符 %c 的 Unicode 码点值为 %d (十六进制: %X)\n", r, r, r)
	fmt.Printf("字符 %c 的 UTF-8 编码长度为 %d 个字节\n", r, utf8.RuneLen(r))

	// 2. 字符串与 rune 的关系
	str := "Hello,世界"
	fmt.Println("\n字符串:", str)
	fmt.Printf("字符串长度(字节数): %d\n", len(str))
	fmt.Printf("字符串长度(Unicode字符数): %d\n", utf8.RuneCountInString(str))

	//每个字符占的字节数
	for i, char := range str {
		fmt.Printf("字符 %d: %c (UTF-8 编码长度: %d 字节)\n", i, char, utf8.RuneLen(char))
	}

	// 3. 遍历字符串中的 Unicode 字符
	fmt.Println("\n按字节遍历字符串可能导致错误:")
	for i := 0; i < len(str); i++ {
		fmt.Printf("%d: %c ", i, str[i])
	}

	fmt.Println("\n\n正确的方式 - 使用 range 遍历 rune:")
	for i, char := range str {
		fmt.Printf("索引 %d: %c (Unicode: %U)\n", i, char, char)
	}

	// 4. 字符串与 rune 切片的转换
	runes := []rune(str)
	fmt.Printf("\n转换为 rune 切片后的长度: %d\n", len(runes))
	fmt.Println("rune 切片中的每个元素:")
	for i, r := range runes {
		fmt.Printf("runes[%d] = %c (Unicode: %U)\n", i, r, r)
	}

	// 5. 从 rune 切片转回字符串
	modifiedRunes := []rune{'你', '好', ',', ' ', 'G', 'o', '!'}
	modifiedStr := string(modifiedRunes)
	fmt.Printf("\n从 rune 切片创建字符串: %s\n", modifiedStr)

	// 6. rune 的实际应用 - 字符串反转
	fmt.Println("\n字符串反转示例:")
	original := "Go语言编程"
	fmt.Printf("原始字符串: %s\n", original)

	// 错误的字节级反转（会破坏UTF-8编码）
	bytes := []byte(original)
	for i, j := 0, len(bytes)-1; i < j; i, j = i+1, j-1 {
		bytes[i], bytes[j] = bytes[j], bytes[i]
	}
	fmt.Printf("错误的字节级反转: %s (乱码)\n", string(bytes))

	// 正确的rune级反转
	runes = []rune(original)
	fmt.Println("before:", runes)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	fmt.Printf("正确的rune级反转: %s\n", string(runes))

	runes2 := []rune(original)
	newRunes := []rune(original)
	for i := 0; i < len(runes2); i++ {
		newRunes[i] = runes2[len(runes2)-1-i]
	}
	fmt.Println("after2:", newRunes)
	fmt.Printf("正确的rune级反转2: %s\n", string(newRunes))

}
