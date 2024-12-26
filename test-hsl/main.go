package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/lucasb-eyer/go-colorful"
)

func main() {
	if len(os.Args) != 5 {
		fmt.Println("Usage: <program> <h> <s> <l> <a>")
		return
	}

	// Parse command-line arguments
	h, _ := strconv.ParseFloat(os.Args[1], 64)
	s, _ := strconv.ParseFloat(os.Args[2], 64)
	l, _ := strconv.ParseFloat(os.Args[3], 64)
	a, _ := strconv.ParseFloat(os.Args[4], 64)

	// Convert HSLA to RGBA
	c := colorful.Hsl(h, s/100.0, l/100.0)
	r, g, b := c.RGB255()
	alpha := uint8(a * 255)

	// Output as RGBA hex string
	fmt.Printf("#%02x%02x%02x%02x\n", r, g, b, alpha)
}
