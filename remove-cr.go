package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

const binaryDetectionBytes = 8000 // Git uses the first 8k bytes for binary detection

func main() {
	// Parse command-line arguments
	write := flag.Bool("w", false, "rewrite files without CR characters")
	flag.Parse()

	// Get list of files, handle Windows-style globbing
	files := expandGlobs(flag.Args())

	if len(files) == 0 {
		fmt.Println("No files provided.")
		os.Exit(1)
	}

	for _, filename := range files {
		// Process each file
		processFile(filename, *write)
	}
}

// Expand glob patterns (for Windows globbing)
func expandGlobs(patterns []string) []string {
	var files []string
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			fmt.Printf("Error in pattern %s: %v\n", pattern, err)
			continue
		}
		files = append(files, matches...)
	}
	return files
}

// Process the file by checking and optionally rewriting
func processFile(filename string, write bool) {
	// Read the entire file as a byte slice
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Printf("Error reading file %s: %v\n", filename, err)
		return
	}

	// Check if the file is binary (like Git does, by checking for null bytes)
	if isBinaryGitStyle(data) {
		fmt.Printf("Skipping binary file: %s\n", filename)
		return
	}

	// Check if there are any CR characters
	if !strings.Contains(string(data), "\r") {
		// No CR found, skip the file
		return
	}

	// Print filename if CRs are found
	fmt.Println(filename)

	// If the -w flag is set, rewrite the file without CRs
	if write {
		newData := strings.ReplaceAll(string(data), "\r", "")
		err := ioutil.WriteFile(filename, []byte(newData), 0644)
		if err != nil {
			fmt.Printf("Error writing to file %s: %v\n", filename, err)
			return
		}
	}
}

// isBinaryGitStyle checks if a file is binary by looking for null bytes in the first 8k bytes
func isBinaryGitStyle(data []byte) bool {
	limit := binaryDetectionBytes
	if len(data) < binaryDetectionBytes {
		limit = len(data)
	}

	// Scan the first `binaryDetectionBytes` for null bytes
	for i := 0; i < limit; i++ {
		if data[i] == 0 {
			return true
		}
	}
	return false
}
