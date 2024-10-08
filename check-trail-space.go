package main

import (
	"bufio"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// hasTrailingWhitespace checks if a line in the file has trailing whitespace.
func hasTrailingWhitespace(filename string) (bool, error) {
	file, err := os.Open(filename)
	if err != nil {
		return false, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	for {
		line, err := reader.ReadString('\n')
		// Trim newline characters, so we can accurately check for trailing spaces or tabs
		trimmedLine := strings.TrimRight(line, "\r\n")

		// Check for trailing whitespace (space or tab)
		if strings.HasSuffix(trimmedLine, " ") || strings.HasSuffix(trimmedLine, "\t") {
			return true, nil
		}

		if err == io.EOF {
			break
		} else if err != nil {
			return false, err
		}
	}

	return false, nil
}

func main() {
	root := "."

	err := filepath.Walk(root, walkFunc)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// walkFunc is the function called for each file in the directory tree.
func walkFunc(path string, info fs.FileInfo, err error) error {
	if err != nil {
		return err
	}

	// Skip directories and symlinks.
	if info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil
	}

	// Check if the file is binary.
	if isBinary(path) {
		return nil
	}

	// Check if the file has trailing whitespace.
	containsWhitespace, err := hasTrailingWhitespace(path)
	if err != nil {
		return err
	}

	if containsWhitespace {
		fmt.Println(path)
	}

	return nil
}
