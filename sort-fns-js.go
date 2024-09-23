package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
)

var fnRe = regexp.MustCompile(`^(export\s+)?function\s+(\w+)`)
var writeBack bool

func main() {
	flag.BoolVar(&writeBack, "w", false, "write modified files back to disk")
	flag.Parse()
	files := flag.Args()
	if len(files) == 0 {
		log.Fatal("No files were specified")
	}
	for _, path := range files {
		if filepath.Ext(path) == ".js" {
			processFile(path)
		}
	}
}

func processFile(path string) {
	// Read the file
	lines := readLines(path)

	// Remember the original contents
	// It is safe to do this by reference here
	// because the new version will be constructed as a new array
	// not by modifying the old array
	old := lines

	// Process the contents
	for _, dent := range indentations(lines) {
		// The input string may or may not contain a JavaScript function declaration
		// Returns the function name if so, otherwise the empty string
		beginFn := func(s string) string {
			s = trimPrefixOrEmpty(s, dent)
			match := fnRe.FindStringSubmatch(s)
			if len(match) > 1 {
				return match[2] // Return the function name 
			}
			return ""
		}

		endFn := func(s string) EndSpecialKind {
			s = trimPrefixOrEmpty(s, dent)
			if s == "}" {
				return endSpecialInclude
			}
			return endSpecialNo
		}

		chunks := parseChunks(isComment, beginFn, endFn, lines)
		sortChunks(chunks)
		specialSpace(chunks)
		lines = joinChunks(chunks)
	}

	// Did anything change?
	if eqStrings(old, lines) {
		return
	}

	// Write results
	if writeBack {
		fmt.Println(path)
		writeLines(path, lines)
	} else {
		printLines(lines)
	}
}
