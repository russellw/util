package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

const carriageReturn = byte(13)

// containsCarriageReturn checks if a file contains the carriage return character.
func containsCarriageReturn(content []byte) bool {
	return bytes.Contains(content, []byte{carriageReturn})
}

// processFile processes a single file, printing its full path if it contains the carriage return character.
func processFile(path string, info os.FileInfo) error {
	if info.IsDir() {
		return nil
	}

	// Check if the file is binary
	if isBinary(path) {
		return nil
	}

	// Read file content
	content, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	// Check for carriage return character
	if containsCarriageReturn(content) {
		fmt.Println(path)
	}

	return nil
}

// walkDir walks the directory tree rooted at the current directory, processing each file.
func walkDir() error {
	return filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		return processFile(path, info)
	})
}

func main() {
	if err := walkDir(); err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}
}
