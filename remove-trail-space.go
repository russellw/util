package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// isBinaryFile detects if a file is binary by first checking its extension
// against a known list of binary types, and then checking for null bytes.
func isBinaryFile(file string) (bool, error) {
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
		return true, nil
	}

	// Open the file for reading
	f, err := os.Open(file)
	if err != nil {
		return false, err
	}
	defer f.Close()

	// Read the first 8000 bytes (like Git does)
	buf := make([]byte, 8000)
	n, err := f.Read(buf)
	if err != nil && err != io.EOF {
		return false, err
	}

	// Check for null bytes in the sample
	if bytes.IndexByte(buf[:n], 0) != -1 {
		return true, nil
	}

	return false, nil
}

// Removes trailing whitespace from a string.
func trimTrailingWhitespace(line string) string {
	return strings.TrimRightFunc(line, unicode.IsSpace)
}

// Processes a file to remove trailing whitespace.
func processFile(file string, writeChanges bool) error {
	isBinary, err := isBinaryFile(file)
	if err != nil {
		return err
	}

	if isBinary {
		return nil
	}

	inputFile, err := os.Open(file)
	if err != nil {
		return err
	}
	defer inputFile.Close()

	var outputLines []string
	changed := false

	scanner := bufio.NewScanner(inputFile)
	for scanner.Scan() {
		line := scanner.Text()
		trimmedLine := trimTrailingWhitespace(line)
		if line != trimmedLine {
			changed = true
		}
		outputLines = append(outputLines, trimmedLine)
	}

	if scanner.Err() != nil {
		return scanner.Err()
	}

	if changed {
		fmt.Printf("Trailing whitespace found in: %s\n", file)
		if writeChanges {
			// Rewriting the file
			err := ioutil.WriteFile(file, []byte(strings.Join(outputLines, "\n")+"\n"), 0644)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// Globs and processes files, checking for binary files and removing trailing whitespace.
func processFiles(files []string, writeChanges bool) error {
	for _, pattern := range files {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return err
		}

		for _, match := range matches {
			err := processFile(match, writeChanges)
			if err != nil {
				fmt.Printf("Error processing file %s: %v\n", match, err)
			}
		}
	}
	return nil
}

func main() {
	writeFlag := flag.Bool("w", false, "Rewrite files with trailing whitespace removed")
	flag.Parse()

	if len(flag.Args()) == 0 {
		fmt.Println("Usage: remove-trail-space [-w] file1 [file2 ...]")
		return
	}

	err := processFiles(flag.Args(), *writeFlag)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	}
}