package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	totalSize := int64(0)

	for scanner.Scan() {
		filePath := scanner.Text()
		fileInfo, err := os.Stat(filePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading file %s: %v\n", filePath, err)
			continue
		}
		fileSize := fileInfo.Size()
		fmt.Printf("%d\t%s\n", fileSize, filePath)
		totalSize += fileSize
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "Error reading input:", err)
	}

	fmt.Printf("%d\n", totalSize)
}
