package main

import (
	"fmt"

	"github.com/google/uuid"
)

func main() {

	//create uuid
	myUUID := uuid.New()
	fmt.Println(myUUID)

}
