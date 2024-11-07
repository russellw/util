package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"regexp"
)

func main() {
	// Check if a filename is provided as a command-line argument
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <filename>")
		return
	}

	// Read the HTML file
	filename := os.Args[1]
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatalf("Error reading file: %v", err)
	}

	// Convert content to string
	htmlContent := string(content)

	// Use regex to remove HTML tags
	re := regexp.MustCompile("<[^>]*>")
	plainText := re.ReplaceAllString(htmlContent, "")

	// Print the plain text
	fmt.Println(plainText)
}
