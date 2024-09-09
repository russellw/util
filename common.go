package main

import (
	"bytes"
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

// isCFamily checks if the given file extension belongs to a language that uses // for single-line comments
func isCFamily(filename string) bool {
	// List of file extensions that correspond to C family languages
	cFamilyExtensions := []string{".c", ".cpp", ".cs", ".java", ".js", ".go", ".h", ".hpp", ".ts"}

	// Get the file extension from the filename
	ext := filepath.Ext(filename)

	// Check if the extension is in the list of C family extensions
	for _, validExt := range cFamilyExtensions {
		if ext == validExt {
			return true
		}
	}
	return false
}
