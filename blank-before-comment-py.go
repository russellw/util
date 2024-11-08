package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
)

// Function to get the indentation of a line
func getIndentation(line string) string {
	return line[:len(line)-len(strings.TrimLeft(line, " \t"))]
}

func main() {
	// Parsing command-line flags and arguments
	writeFlag := flag.Bool("w", false, "Write changes back to the file")
	flag.Parse()

	// Check if a source file is provided as argument
	if len(flag.Args()) < 1 {
		fmt.Println("Usage: go run main.go [-w] <sourcefile>")
		return
	}

	// Open the input file
	filePath := flag.Args()[0]
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Printf("Error opening file: %v\n", err)
		return
	}
	defer file.Close()

	// Read the content of the file
	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		return
	}

	// Process lines to add blank lines before comments
	var result []string
	for i := 0; i < len(lines); i++ {
		line := lines[i]

		// Detect comment lines (lines that start with #)
		if strings.TrimSpace(line) != "" && strings.HasPrefix(strings.TrimSpace(line), "#") {
			// Check if it's preceded by a non-blank, non-comment line with same indentation
			if i > 0 && strings.TrimSpace(lines[i-1]) != "" && !strings.HasPrefix(strings.TrimSpace(lines[i-1]), "#") {
				// Check indentation of the previous line
				prevIndent := getIndentation(lines[i-1])
				currIndent := getIndentation(line)
				if prevIndent == currIndent {
					// Insert a blank line before the comment
					result = append(result, "")
				}
			}
		}
		// Add the current line to the result
		result = append(result, line)
	}

	// Write the result to stdout or overwrite the file
	if *writeFlag {
		// Write to the original file
		file, err := os.Create(filePath)
		if err != nil {
			fmt.Printf("Error opening file for writing: %v\n", err)
			return
		}
		defer file.Close()

		for _, line := range result {
			file.WriteString(line + "\n")
		}
	} else {
		// Output to stdout
		for _, line := range result {
			fmt.Println(line)
		}
	}
}
