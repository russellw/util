package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"strings"
)

func main() {
	// Parse the -w flag
	write := flag.Bool("w", false, "write result to file instead of printing")
	flag.Parse()

	// Get the list of files from positional arguments
	files := flag.Args()

	if len(files) == 0 {
		fmt.Println("Usage: comment-space [-w] <files>")
		return
	}

	// Process each file
	for _, file := range files {
		processFile(file, *write)
	}
}

func processFile(filename string, write bool) {
	if !isCFamily(filename) {
		return
	}

	data, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal(err)
	}

	lines := strings.Split(string(data), "\n")
	changed := false
	var output []string

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if len(trimmed) > 2 && strings.HasPrefix(trimmed, "//") && !strings.HasPrefix(trimmed, "// ") {
			fmt.Printf("%s:%d: %s\n", filename, i+1, line)
			changed = true
			line = strings.Replace(line, "//", "// ", 1)
		}
		output = append(output, line)
	}

	if changed && write {
		// In this case it is correct to not explicitly re-add the trailing newline
		// because the lines were obtained with strings.Split
		err = ioutil.WriteFile(filename, []byte(strings.Join(output, "\n")), 0644)
		if err != nil {
			log.Fatal(err)
		}
	}
}
