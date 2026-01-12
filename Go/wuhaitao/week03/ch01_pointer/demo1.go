package main

import "fmt"

type Point struct {
	x int
	y int
}

func changePoint(p *Point, scale float64) {
	p.x = int(float64(p.x) * scale)
	p.y = int(float64(p.y) * scale)
}

func changePoint2(p Point, scale float64) {
	p.x = int(float64(p.x) * scale)
	p.y = int(float64(p.y) * scale)
}
func main() {

	p := Point{10, 2}

	changePoint2(p, 2.1)
	fmt.Println(p)

	changePoint(&p, 2.1)
	fmt.Println(p)

}
