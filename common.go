package main

import (
	"bytes"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
)

var ignored = map[string]struct{}{
	".git": {}, "node_modules": {}, "__pycache__": {},
}

func ignore(dir string) bool {
	_, exists := ignored[dir]
	return exists
}

// Known binary file extensions
var binaryExtensions = map[string]struct{}{
	".pdf": {}, ".png": {}, ".exe": {}, ".jpg": {}, ".jpeg": {},
	".gif": {}, ".bmp": {}, ".zip": {}, ".rar": {}, ".tar": {},
	".gz": {}, ".7z": {}, ".dll": {}, ".iso": {}, ".mp3": {},
	".mp4": {}, ".avi": {}, ".mkv": {}, ".mov": {}, ".bin": {},
	".dmg": {}, ".class": {}, ".so": {}, ".o": {}, ".obj": {},
}

// isBinary detects if a file is binary by first checking its extension
// against a known list of binary types, and then checking the entire file for null bytes.
func isBinary(file string) bool {
	// Get file extension and convert to lowercase for case-insensitive comparison
	ext := strings.ToLower(filepath.Ext(file))
	_, exists := binaryExtensions[ext]
	if exists {
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
