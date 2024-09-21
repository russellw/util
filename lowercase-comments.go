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

func exempt(text string) bool {
	return unicode.IsUpper(rune(text[1]))
}

func lowercaseFirstWord(comment string) string {
	if exempt(comment) {
		return comment
	}
	runes := []rune(comment)
	for i, r := range runes {
		if unicode.IsLetter(r) {
			runes[i] = unicode.ToLower(r)
			break
		}
	}
	return string(runes)
}

func main() {
	writeBack := flag.Bool("w", false, "write result to the file")
	flag.Parse()

	if flag.NArg() == 0 {
		log.Fatal("No files provided")
	}

	var files []string
	for _, arg := range flag.Args() {
		matches, err := filepath.Glob(arg)
		if err != nil {
			log.Fatal(err)
		}
		files = append(files, matches...)
	}

	if len(files) == 0 {
		fmt.Println("No matching files found")
		os.Exit(1)
	}

	for _, fileName := range files {
		processFile(fileName, *writeBack)
	}
}

// Processes a single file
func processFile(fileName string, writeBack bool) {
	if !isCFamily(fileName) {
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
		blockStart  int
	)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, "// ") {
			comment := strings.TrimPrefix(trimmed, "// ")

			blockStart = len(lines) - 1

			// Extract leading whitespace
			leadingWhitespace := line[:strings.Index(line, "// ")]

			firstWord := strings.Fields(comment)[0]
			if !exempt(firstWord) && unicode.IsUpper(rune(firstWord[0])) {
				fmt.Printf("%s:%d: %s\n", fileName, blockStart+1, line)
				lowercased := lowercaseFirstWord(comment)
				lines[blockStart] = leadingWhitespace + "// " + lowercased
				needsFixing = true
			}
		}
	}

	if writeBack && needsFixing {
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
