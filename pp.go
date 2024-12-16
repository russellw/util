package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	definePattern = regexp.MustCompile(`^#define\s+(\w+)\s+(.+)$`)
	includePattern = regexp.MustCompile(`^#include\s+"([^"]+)"$`)
)

type Preprocessor struct {
	defines map[string]string
}

func NewPreprocessor() *Preprocessor {
	return &Preprocessor{
		defines: make(map[string]string),
	}
}

func isAbsolutePath(path string) bool {
	return strings.HasPrefix(path, "/") || strings.HasPrefix(path, "\\")
}

func (p *Preprocessor) ProcessFile(filename string) string {
	file, err := os.Open(filename)
	if err != nil {
		panic(fmt.Errorf("failed to open file %s: %w", filename, err))
	}
	defer file.Close()

	var result strings.Builder
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if defineMatch := definePattern.FindStringSubmatch(line); defineMatch != nil {
			// Handle #define
			key := defineMatch[1]
			value := defineMatch[2]
			p.defines[key] = value
		} else if includeMatch := includePattern.FindStringSubmatch(line); includeMatch != nil {
			// Handle #include
			includeFile := includeMatch[1]
			if !isAbsolutePath(includeFile) {
				includeFile = filepath.Join(filepath.Dir(filename), includeFile)
			}
			includedContent := p.ProcessFile(includeFile)
			result.WriteString(includedContent)
		} else {
			// Replace defined symbols in the line
			for key, value := range p.defines {
				line = replaceSymbols(line, key, value)
			}
			result.WriteString(line + "\n")
		}
	}

	if err := scanner.Err(); err != nil {
		panic(fmt.Errorf("error reading file %s: %w", filename, err))
	}

	return result.String()
}

func replaceSymbols(line, key, value string) string {
	// Ensure we only replace whole words by using word boundaries
	pattern := regexp.MustCompile(`\b` + regexp.QuoteMeta(key) + `\b`)
	return pattern.ReplaceAllString(line, value)
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: go run main.go <input file>")
		os.Exit(1)
	}

	inputFile := os.Args[1]
	preprocessor := NewPreprocessor()
	result := preprocessor.ProcessFile(inputFile)
	fmt.Print(result)
}
