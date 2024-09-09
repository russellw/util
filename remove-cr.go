package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"strings"
)

func main() {
	// Parse command-line arguments
	write := flag.Bool("w", false, "rewrite files without CR characters")
	flag.Parse()

	// Get list of files
	files := flag.Args()

	if len(files) == 0 {
		log.Fatal("No files provided.")
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
		log.Fatal(err)
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
			log.Fatal(err)
		}
	}
}
