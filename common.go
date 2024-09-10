package main

import (
	"bufio"
	"bytes"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type Chunk struct {
	name  string
	lines []string
}

type EndSpecialKind int

const (
	endSpecialNo EndSpecialKind = iota
	endSpecialExclude
	endSpecialInclude
)

var commentRe = regexp.MustCompile(`\s*//`)

// Known binary file extensions
var binaryExtensions = map[string]struct{}{
	".pdf": {}, ".png": {}, ".exe": {}, ".jpg": {}, ".jpeg": {},
	".gif": {}, ".bmp": {}, ".zip": {}, ".rar": {}, ".tar": {},
	".gz": {}, ".7z": {}, ".dll": {}, ".iso": {}, ".mp3": {},
	".mp4": {}, ".avi": {}, ".mkv": {}, ".mov": {}, ".bin": {},
	".dmg": {}, ".class": {}, ".so": {}, ".o": {}, ".obj": {},
}

var ignored = map[string]struct{}{
	".git": {}, "node_modules": {}, "__pycache__": {},
}

func appendChunk(chunks []Chunk, name string, lines []string) []Chunk {
	if len(lines) == 0 {
		return chunks
	}
	chunk := Chunk{name: name, lines: lines}
	return append(chunks, chunk)
}

func parseChunks(isComment func(string) bool, beginSpecial func(string) string, endSpecial func(string) EndSpecialKind, lines []string) []Chunk {
	n := len(lines)
	var chunks []Chunk
	for i := 0; i < n; {
		// non-special chunk?
		j := i
		for j < n && beginSpecial(lines[j]) == "" {
			j++
		}
		k := j
		for i < j && isComment(lines[j-1]) {
			j--
		}
		chunks = appendChunk(chunks, "", lines[i:j])
		i = j

		// special chunk?
		if i == n {
			break
		}
		name := beginSpecial(lines[k])
	loop:
		for j = k + 1; j < n; j++ {
			switch endSpecial(lines[j]) {
			case endSpecialExclude:
				break loop
			case endSpecialInclude:
				j++
				break loop
			}
		}
		chunks = appendChunk(chunks, name, lines[i:j])
		i = j
	}
	return chunks
}

func isComment(s string) bool {
	return commentRe.MatchString(s)
}

func trimBlankLines(lines []string) []string {
	// Iterate from the end of the slice and find the first non-blank line
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.TrimSpace(lines[i]) != "" {
			// Return the slice up to and including the last non-blank line
			return lines[:i+1]
		}
	}
	// If all lines are blank, return an empty slice
	return []string{}
}

func ignore(dir string) bool {
	_, exists := ignored[dir]
	return exists
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
func isCFamily(path string) bool {
	// List of file extensions that correspond to C family languages
	cFamilyExtensions := []string{".c", ".cpp", ".cs", ".java", ".js", ".go", ".h", ".hpp", ".ts"}

	// Get the file extension from the path
	ext := filepath.Ext(path)

	// Check if the extension is in the list of C family extensions
	for _, validExt := range cFamilyExtensions {
		if ext == validExt {
			return true
		}
	}
	return false
}

func readLines(path string) []string {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	return lines
}

func writeLines(path string, lines []string) {
	// Create or open the file for writing
	file, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Create a new writer
	writer := bufio.NewWriter(file)
	for _, line := range lines {
		// Write the line without concatenating the newline
		_, err := writer.WriteString(line)
		if err != nil {
			log.Fatal(err)
		}

		// Write the newline separately
		_, err = writer.WriteString("\n")
		if err != nil {
			log.Fatal(err)
		}
	}

	// Flush any buffered data to the file
	err = writer.Flush()
	if err != nil {
		log.Fatal(err)
	}
}
