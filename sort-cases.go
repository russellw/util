package main

import (
	"flag"
	"fmt"
	"log"
	"regexp"
	"sort"
)

var caseRe = regexp.MustCompile(`^\s+case `)
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
	old := copySlice(lines)

	// Process the contents
	for i := 0; ; {
		i = findString(caseRe, lines, i)
		if i < 0 {
			break
		}
		j := findStringNot(caseRe, lines, i+1)
		if j < 0 {
			log.Fatalf("%s:%d: unexpected end of file", path, i+1)
		}
		sort.Strings(lines[i:j])
		i = j
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
