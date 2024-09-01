package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	lineCount := 0

	for scanner.Scan() {
		line := scanner.Text()
		fmt.Println(line)
		lineCount++
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "Error reading input:", err)
	}

	fmt.Printf("%d\n", lineCount)
}
