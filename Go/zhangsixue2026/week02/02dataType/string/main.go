package main

import (
	"fmt"
	"strconv"
	"strings"
	"unsafe"
)

func main() {
	fmt.Println("========== 一、字符串的定义与基本特性 ==========")

	// 1. 双引号字符串
	s1 := "hello"
	s2 := "你好"
	fmt.Println("双引号字符串:", s1, s2)

	// 2. 多行字符串（反引号）
	s3 := `第一行
	第二行
	第三行`
	fmt.Println("多行字符串:", s3)

	// 3. 字符串转义符
	s4 := "这是第一行\n这是第二行"
	fmt.Println("换行符示例:")
	fmt.Println(s4)

	s5 := "Tab:\t缩进"
	fmt.Println(s5)

	s6 := "C:\\Windows\\System32"
	fmt.Println("路径示例:", s6)

	fmt.Println("\n========== 二、字符类型：byte 与 rune ==========")

	// 1. byte 类型（uint8别名）
	var a byte = 'A'
	var b uint8 = 66
	fmt.Printf("byte类型: a=%c(%d), b=%c(%d)\n", a, a, b, b)

	// 2. rune 类型（int32别名）
	var chineseChar rune = '中'
	fmt.Printf("rune类型: %c\n", chineseChar)

	// 3. byte和rune占用字节数对比
	var byteVar byte = 'A'
	var runeVar rune = 'B'
	fmt.Printf("byte占用 %d 个字节\n", unsafe.Sizeof(byteVar))
	fmt.Printf("rune占用 %d 个字节\n", unsafe.Sizeof(runeVar))

	fmt.Println("\n========== 三、字符串的遍历与长度 ==========")

	// 1. 字符串长度（字节数）
	s7 := "hello,中国"
	fmt.Printf("字符串 '%s' 的字节长度: %d\n", s7, len(s7))
	fmt.Printf("字符串 '%s' 的字符数量: %d\n", s7, len([]rune(s7)))

	// 2. 按字节遍历（会出现乱码）
	fmt.Println("\n按字节遍历:")
	s8 := "你好"
	for i := 0; i < len(s8); i++ {
		fmt.Printf("%v(%c) ", s8[i], s8[i])
	}
	fmt.Println()

	// 3. 按字符遍历（推荐）
	fmt.Println("\n按字符遍历（使用 for range）:")
	s9 := "你好，世界"
	for index, char := range s9 {
		fmt.Printf("索引:%d, 字符:%c\n", index, char)
	}

	fmt.Println("\n========== 四、字符串的修改与类型转换 ==========")

	// 1. 修改英文字符串（使用[]byte）
	s10 := "hello"
	byteSlice := []byte(s10)
	byteSlice[0] = 'H'
	newS10 := string(byteSlice)
	fmt.Printf("修改前: %s, 修改后: %s\n", s10, newS10)

	// 2. 修改中文字符串（使用[]rune）
	s11 := "博客"
	runeSlice := []rune(s11)
	runeSlice[0] = '狗'
	newS11 := string(runeSlice)
	fmt.Printf("修改前: %s, 修改后: %s\n", s11, newS11)

	// 3. 数值转字符串
	num := 42
	str := strconv.Itoa(num)
	fmt.Printf("int转string: %s, 类型: %T\n", str, str)

	// FormatInt 示例
	hexStr := strconv.FormatInt(15, 16)
	fmt.Printf("15的16进制字符串: %s\n", hexStr)

	// FormatFloat 示例
	floatStr := strconv.FormatFloat(3.14159, 'f', 2, 64)
	fmt.Printf("浮点数转字符串: %s\n", floatStr)

	// 4. 字符串转数值
	s12 := "12345"
	i, err := strconv.Atoi(s12)
	if err != nil {
		fmt.Println("转换失败:", err)
	} else {
		fmt.Printf("string转int: %d, 类型: %T\n", i, i)
	}

	// ParseInt 示例
	i64, err := strconv.ParseInt("ff", 16, 64)
	if err == nil {
		fmt.Printf("16进制'ff'转十进制: %d\n", i64)
	}

	// ParseFloat 示例
	f64, err := strconv.ParseFloat("3.14", 64)
	if err == nil {
		fmt.Printf("字符串转浮点数: %f\n", f64)
	}

	fmt.Println("\n========== 五、字符串的常用操作 ==========")

	// 1. 字符串拼接
	str1 := "Go"
	str2 := "lang"
	fmt.Println("字符串拼接:", str1+str2)

	// 2. 字符串分割
	csvStr := "a,b,c"
	parts := strings.Split(csvStr, ",")
	fmt.Printf("分割字符串: %v\n", parts)

	// 3. 字符串连接
	joined := strings.Join([]string{"a", "b", "c"}, "-")
	fmt.Printf("连接字符串: %s\n", joined)

	// 4. 判断包含
	fmt.Printf("'golang'包含'go': %t\n", strings.Contains("golang", "go"))

	// 5. 前缀判断
	fmt.Printf("'hello'以'he'开头: %t\n", strings.HasPrefix("hello", "he"))

	// 6. 后缀判断
	fmt.Printf("'hello'以'lo'结尾: %t\n", strings.HasSuffix("hello", "lo"))

	// 7. 查找子串位置
	index := strings.Index("golang", "la")
	fmt.Printf("'la'在'golang'中的位置: %d\n", index)

	// 8. 替换字符串
	replaced := strings.Replace("hello hello", "hello", "hi", 1)
	fmt.Printf("替换一次: %s\n", replaced)

	replacedAll := strings.ReplaceAll("hello hello", "hello", "hi")
	fmt.Printf("替换全部: %s\n", replacedAll)

	// 9. 大小写转换
	fmt.Printf("转大写: %s\n", strings.ToUpper("hello"))
	fmt.Printf("转小写: %s\n", strings.ToLower("HELLO"))

	// 10. 去除空格
	trimmed := strings.TrimSpace("  hello  ")
	fmt.Printf("去除首尾空格: '%s'\n", trimmed)

	// 11. 格式化字符串
	formatted := fmt.Sprintf("Go版本%d.%d", 1, 21)
	fmt.Printf("格式化字符串: %s\n", formatted)

	fmt.Println("\n========== 六、字符串拼接效率对比（使用strings.Builder） ==========")

	// 使用 strings.Builder 高效拼接
	var builder strings.Builder
	builder.Grow(100) // 预估容量，减少内存重分配

	parts2 := []string{"Hello", " ", "from", " ", "Go", "!"}
	for _, p := range parts2 {
		builder.WriteString(p)
	}
	result := builder.String()
	fmt.Printf("使用Builder拼接: %s\n", result)

	fmt.Println("\n========== 七、字符串的格式化输出 ==========")

	path := "C:\\temp\\file.txt"

	// %s: 直接输出
	fmt.Printf("路径原始输出: %s\n", path)

	// %q: 输出解释型字符串（带引号和转义）
	fmt.Printf("路径带引号输出: %q\n", path)

	// %c: 打印字符
	fmt.Printf("第一个字符: %c\n", path[0])

	// %v: 默认格式
	fmt.Printf("默认格式: %v\n", path)

	// %#v: Go语法表示
	fmt.Printf("Go语法表示: %#v\n", path)

	fmt.Println("\n========== 八、实用示例 ==========")
}
