package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/fs"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
)

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

func main() {
	root := "."

	err := filepath.Walk(root, walkFunc)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
