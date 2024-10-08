package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// Capitalizes the first word of the comment if needed
func capitalizeFirstWord(comment string) string {
	if isURL(comment) {
		return comment
	}
	runes := []rune(comment)
	for i, r := range runes {
		if unicode.IsLetter(r) {
			runes[i] = unicode.ToUpper(r)
			break
		}
	}
	return string(runes)
}

// Checks if the string is a URL
func isURL(text string) bool {
	return strings.HasPrefix(text, "http://") || strings.HasPrefix(text, "https://")
}

func main() {
	writeBack := flag.Bool("w", false, "write result to the file")
	flag.Parse()

	if flag.NArg() == 0 {
		log.Fatal("No files provided")
	}

	files := flag.Args()

	for _, fileName := range files {
		processFile(fileName, *writeBack)
	}
}

// Processes a single file
func processFile(fileName string, writeBack bool) {
	if filepath.Ext(fileName) != ".py" {
		return
	}

	file, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	var (
		lines       []string
		needsFixing bool
		inBlock     bool
		blockStart  int
	)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, "# ") {
			comment := strings.TrimPrefix(trimmed, "# ")

			if !inBlock {
				inBlock = true
				blockStart = len(lines) - 1

				// Extract leading whitespace
				leadingWhitespace := line[:strings.Index(line, "# ")]

				// Capitalize the first word if necessary
				firstWord := strings.Fields(comment)[0]
				if !isURL(firstWord) && unicode.IsLower(rune(firstWord[0])) {
					fmt.Printf("%s:%d: %s\n", fileName, blockStart+1, line)
					capitalized := capitalizeFirstWord(comment)
					lines[blockStart] = leadingWhitespace + "# " + capitalized
					needsFixing = true
				}
			}
		} else {
			inBlock = false
		}
	}

	if writeBack && needsFixing {
		fmt.Println(fileName)

		// Write back to the file
		file, err := os.Create(fileName)
		if err != nil {
			log.Fatal(err)
		}
		defer file.Close()

		writer := bufio.NewWriter(file)
		for _, line := range lines {
			_, err := writer.WriteString(line + "\n")
			if err != nil {
				log.Fatal(err)
			}
		}
		writer.Flush()
	}
}
