package main

import (
	"fmt"
	"strings"
)

func main() {
	// 基本数据类型
	var integerVar int = 42
	var floatVar float64 = 3.14
	var boolVar bool = true
	var stringVar string = "Hello, Go!"

	// 简短声明
	shortInt := 100
	shortString := "简短声明"

	// 打印变量值和类型
	fmt.Println("=== 基本数据类型 ===")
	fmt.Printf("整数: %v, 类型: %T\n", integerVar, integerVar)
	fmt.Printf("浮点数: %v, 类型: %T\n", floatVar, floatVar)
	fmt.Printf("布尔值: %v, 类型: %T\n", boolVar, boolVar)
	fmt.Printf("字符串: %v, 类型: %T\n", stringVar, stringVar)
	fmt.Printf("简短声明整数: %v\n", shortInt)
	fmt.Printf("简短声明字符串: %v\n", shortString)

	// 字符串操作
	fmt.Println("\n=== 字符串操作 ===")

	// 字符串连接
	firstName := "张"
	lastName := "三"
	fullName := firstName + lastName
	fmt.Printf("字符串连接: %s + %s = %s\n", firstName, lastName, fullName)

	// 使用fmt.Sprintf连接字符串
	formattedString := fmt.Sprintf("姓名: %s, 年龄: %d", fullName, 30)
	fmt.Printf("格式化字符串: %s\n", formattedString)

	// 字符串长度
	text := "你好，世界！"
	fmt.Printf("字符串 '%s' 的字节长度: %d\n", text, len(text))
	fmt.Printf("字符串 '%s' 的字符数量: %d\n", text, len([]rune(text)))

	// 字符串索引和切片
	message := "Go编程很有趣"
	fmt.Printf("字符串: %s\n", message)
	fmt.Printf("第一个字符: %s\n", string([]rune(message)[0]))
	fmt.Printf("前两个字符: %s\n", string([]rune(message)[:2]))

	// 字符串包含判断
	sentence := "Go语言是一种编译型语言"
	fmt.Printf("'%s' 是否包含 '语言': %v\n", sentence, strings.Contains(sentence, "语言"))
	fmt.Printf("'%s' 是否包含 'Java': %v\n", sentence, strings.Contains(sentence, "Java"))

	// 字符串替换
	oldText := "Hello, World!"
	newText := strings.Replace(oldText, "World", "Go", 1)
	fmt.Printf("替换前: %s\n", oldText)
	fmt.Printf("替换后: %s\n", newText)

	// 字符串分割
	csvData := "苹果,香蕉,橙子,葡萄"
	fruits := strings.Split(csvData, ",")
	fmt.Printf("分割字符串 '%s':\n", csvData)
	for i, fruit := range fruits {
		fmt.Printf("  水果[%d]: %s\n", i, fruit)
	}

	// 字符串转换大小写
	mixedCase := "Hello Go World"
	fmt.Printf("原始字符串: %s\n", mixedCase)
	fmt.Printf("转大写: %s\n", strings.ToUpper(mixedCase))
	fmt.Printf("转小写: %s\n", strings.ToLower(mixedCase))
}
