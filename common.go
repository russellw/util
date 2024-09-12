package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
)

type Range struct {
	i, j int
}

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
	chunk := Chunk{name: name, lines: copySlice(lines)}
	return append(chunks, chunk)
}

func appendUnique(v []string, s string) []string {
	if contains(v, s) {
		return v
	}
	return append(v, s)
}

func contains(slice []string, str string) bool {
	for _, v := range slice {
		if v == str {
			return true
		}
	}
	return false
}

func copySlice(v []string) []string {
	r := make([]string, len(v))
	copy(r, v)
	return r
}

// Compare slices of string, accounting for nil vs empty slice differences
func eqStrings(a, b []string) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

// Find the first string that matches the regular expression, starting the search at the given index
// and return the found index
// or -1 if there was no occurrence
func findString(re *regexp.Regexp, v []string, i int) int {
	for ; i < len(v); i++ {
		if re.MatchString(v[i]) {
			return i
		}
	}
	return -1
}

func getRanges[T any](f func(T) bool, v []T) []Range {
	n := len(v)
	var ranges []Range
	for i := 0; i < n; {
		// negative range?
		for i < n && !f(v[i]) {
			i++
		}

		// positive range?
		if i == n {
			break
		}
		j := i + 1
		for ; j < n; j++ {
			if !f(v[j]) {
				break
			}
		}
		ranges = append(ranges, Range{i: i, j: j})
		i = j
	}
	return ranges
}

func ignore(dir string) bool {
	_, exists := ignored[dir]
	return exists
}

// Return the longest prefix of the string, that consists of indentation characters
func indentation(s string) string {
	var i int
	for i = 0; i < len(s); i++ {
		if s[i] != ' ' && s[i] != '\t' {
			break
		}
	}
	return s[:i]
}

// Return the set of indentations of all strings in the slice, represented as an array
func indentations(v []string) []string {
	var r []string
	for _, s := range v {
		r = appendUnique(r, indentation(s))
	}
	return r
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

func isComment(s string) bool {
	return commentRe.MatchString(s)
}

// Join the lines of each chunk together, into a single large array of text lines
func joinChunks(chunks []Chunk) []string {
	var result []string
	for _, chunk := range chunks {
		result = append(result, chunk.lines...)
	}
	return result
}

func parseChunks(isComment func(string) bool, beginSpecial func(string) string, endSpecial func(string) EndSpecialKind, lines []string) []Chunk {
	n := len(lines)
	var chunks []Chunk
	for i := 0; i < n; {
		// Non-special chunk?
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

		// Special chunk?
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

				// If the end marker is included
				// then it could be followed by blank lines which can also be included
				for ; j < n && lines[j] == ""; j++ {
				}
				break loop
			}
		}
		chunks = appendChunk(chunks, name, lines[i:j])
		i = j
	}
	return chunks
}

func printLines(lines []string) {
	for _, line := range lines {
		fmt.Println(line)
	}
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

func sortChunks(chunks []Chunk) {
	for _, r := range specialRanges(chunks) {
		sort.SliceStable(chunks[r.i:r.j], func(i, j int) bool {
			return chunks[r.i+i].name < chunks[r.i+j].name
		})
	}
}

func special(chunk Chunk) bool {
	return chunk.name != ""
}

func specialRanges(chunks []Chunk) []Range {
	return getRanges(special, chunks)
}

func specialSpace(chunks []Chunk) {
	for i := range chunks {
		if !special(chunks[i]) {
			continue
		}
		chunks[i].lines = trimBlankLines(chunks[i].lines)
		if i < len(chunks)-1 {
			chunks[i].lines = append(chunks[i].lines, "")
		}
	}
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
