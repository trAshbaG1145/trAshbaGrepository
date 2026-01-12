package main

import (
	"fmt"
	"strconv"
)

// 定义一个接口
type Good interface {
	settleAccount() int
	orderInfo() string
}

type Phone struct {
	name     string
	quantity int
	price    int
}

func (phone Phone) settleAccount() int {
	return phone.quantity * phone.price
}
func (phone Phone) orderInfo() string {
	return "您要购买" + strconv.Itoa(phone.quantity) + "个" +
		phone.name + "计：" + strconv.Itoa(phone.settleAccount()) + "元"
}

type FreeGift struct {
	name     string
	quantity int
	price    int
}

func (gift FreeGift) settleAccount() int {
	return 0
}
func (gift FreeGift) orderInfo() string {
	return "您要购买" + strconv.Itoa(gift.quantity) + "个" +
		gift.name + "计：" + strconv.Itoa(gift.settleAccount()) + "元"
}

func calculateAllPrice(goods []Good) int {
	var allPrice int
	for _, good := range goods {
		fmt.Println(good.orderInfo())
		allPrice += good.settleAccount()
	}
	return allPrice
}
func main() {
	// demo1
	// iPhone := Phone{
	// 	name:     "iPhone",
	// 	quantity: 1,
	// 	price:    8000,
	// }
	// earphones := FreeGift{
	// 	name:     "耳机",
	// 	quantity: 1,
	// 	price:    200,
	// }

	// goods := []Good{iPhone, earphones}
	// allPrice := calculateAllPrice(goods)
	// fmt.Printf("该订单总共需要支付 %d 元", allPrice)

	// demo2

	// 声明一个空接口实例
	var i interface{}

	// 存 int 没有问题
	i = 1
	fmt.Println(i)

	// 存字符串也没有问题
	i = "hello"
	fmt.Println(i)

	// 存布尔值也没有问题
	i = false
	fmt.Println(i)
}
