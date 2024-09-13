package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
)

var caseRe = regexp.MustCompile(`^case `)
var writeBack bool

func main() {
	flag.BoolVar(&writeBack, "w", false, "write modified files back to disk")
	flag.Parse()
	files := flag.Args()
	if len(files) == 0 {
		log.Fatal("No files were specified")
	}
	for _, path := range files {
		if isCFamily(path) {
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
		begin := func(s string) string {
			s = trimPrefixOrEmpty(s, dent)
			if caseRe.MatchString(s) {
				return s
			}
			return ""
		}

		end := func(s string) EndSpecialKind {
			if indentation(s) == dent {
				return endSpecialExclude
			}
			return endSpecialNo
		}

		chunks := parseChunks(isComment, begin, end, lines)
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
