package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const carriageReturn = byte(13)

// isBinary detects if a file is binary by first checking its extension
// against a known list of binary types, and then checking the entire file for null bytes.
func isBinary(file string) bool {
	// Known binary file extensions
	binaryExtensions := map[string]bool{
		".pdf": true, ".png": true, ".exe": true, ".jpg": true, ".jpeg": true,
		".gif": true, ".bmp": true, ".zip": true, ".rar": true, ".tar": true,
		".gz": true, ".7z": true, ".dll": true, ".iso": true, ".mp3": true,
		".mp4": true, ".avi": true, ".mkv": true, ".mov": true, ".bin": true,
		".dmg": true, ".class": true, ".so": true, ".o": true, ".obj": true,
	}

	// Get file extension and convert to lowercase for case-insensitive comparison
	ext := strings.ToLower(filepath.Ext(file))
	if binaryExtensions[ext] {
		return true
	}

	// Open the file for reading
	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err) // Crash the program on error
	}
	defer f.Close()

	// Read the entire contents of the file
	data, err := ioutil.ReadAll(f)
	if err != nil {
		log.Fatal(err) // Crash the program on error
	}

	// Check for null bytes in the entire file
	if bytes.IndexByte(data, 0) != -1 {
		return true
	}

	return false
}

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
