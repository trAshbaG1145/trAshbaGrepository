package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"time"
)

// fmt
func fmtDemo() {
	s := "枯藤abc"
	fmt.Printf("%s\n", s)
	fmt.Printf("%q\n", s)
	fmt.Printf("%x\n", s)
	fmt.Printf("%X\n", s)
	var (
		name        string
		age         int
		yanjiusheng bool
	)
	fmt.Scan(&name, &age, &yanjiusheng)
	fmt.Printf("name:%s age:%d married:%t \n", name, age, yanjiusheng)
}

// timeDemo
func timeDemo() {
	t := time.Now()
	fmt.Println(t) //获取当前时间
	t1 := t.Unix() //时间戳
	fmt.Println(t1)
	fmt.Println(t.Format("2006-01-02 15:04:05"))
	j := 0
	for i := 0; i < 1000000000; i++ {
		j++
	}
	fmt.Println(time.Since(t))
}

// flagDemo

func flagDemo1() {
	if len(os.Args) > 0 {
		for index, arg := range os.Args {
			fmt.Printf("args[%d]=%v\n", index, arg)
		}
	}
}

func flagDemo2() {
	//定义命令行参数方式1
	var name string
	var age int
	var yanjiusheng bool
	flag.StringVar(&name, "name", "张三", "姓名")
	flag.IntVar(&age, "age", 18, "年龄")
	flag.BoolVar(&yanjiusheng, "yanjiusheng", false, "婚否")

	//解析命令行参数
	flag.Parse()
	fmt.Println(name, age, yanjiusheng)
	//返回命令行参数后的其他参数
	fmt.Println(flag.Args())
	fmt.Println(flag.NArg())
	fmt.Println(flag.NFlag())

}

// IOdemo
func re() {
	content, err := ioutil.ReadFile("../demo05/main.go")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(content))
}

// strconv

func strconvDemo() {
	i := 122
	fmt.Println(strconv.Itoa(i))

	s := "223"
	i1, err := strconv.Atoi(s)
	fmt.Println(i1)
	fmt.Println(err)

	i2, _ := strconv.ParseBool("s")
	fmt.Println(i2)
}

func main() {
	// fmtDemo()
	// timeDemo()
	// flagDemo2()
	// re()
	// strconvDemo()

	str1 := "abc这发 阿范 德萨"
	str2 := []byte(str1)
	fmt.Println(len(str2))

	// strings.Split(str1, "")
	fmt.Println(len(strings.Split(str1, " ")))
}
