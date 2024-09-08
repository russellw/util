package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	// Parse command-line arguments
	write := flag.Bool("w", false, "rewrite files without CR characters")
	flag.Parse()

	// Get list of files, handle Windows-style globbing
	files := flag.Args()

	if len(files) == 0 {
		fmt.Println("No files provided.")
		os.Exit(1)
	}

	for _, filename := range files {
		// Process each file
		processFile(filename, *write)
	}
}

// Process the file by checking and optionally rewriting
func processFile(filename string, write bool) {
	// Check if the file is binary
	if isBinary(filename) {
		return
	}

	// Read the entire file as a byte slice
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Printf("Error reading file %s: %v\n", filename, err)
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
