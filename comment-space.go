package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strings"
)

func main() {
	// Parse the -w flag
	write := flag.Bool("w", false, "write result to file instead of printing")
	flag.Parse()

	// Get the list of files from positional arguments
	files := flag.Args()

	if len(files) == 0 {
		fmt.Println("Usage: go run main.go [-w] <files>")
		return
	}

	// Process each file
	for _, pattern := range files {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			fmt.Printf("Failed to process pattern %s: %v\n", pattern, err)
			continue
		}

		for _, file := range matches {
			processFile(file, *write)
		}
	}
}

func processFile(filename string, write bool) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Printf("Failed to read file %s: %v\n", filename, err)
		return
	}

	lines := strings.Split(string(data), "\n")
	changed := false
	var output []string

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "//") && !strings.HasPrefix(trimmed, "// ") {
			fmt.Printf("%s:%d: %s\n", filename, i+1, line)
			changed = true
			line = strings.Replace(line, "//", "// ", 1)
		}
		output = append(output, line)
	}

	if changed && write {
		err = ioutil.WriteFile(filename, []byte(strings.Join(output, "\n")), 0644)
		if err != nil {
			fmt.Printf("Failed to write file %s: %v\n", filename, err)
		}
	}
}
