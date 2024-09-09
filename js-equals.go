package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
)

func main() {
	// Parse the -w flag
	writeFlag := flag.Bool("w", false, "write modified files back to disk")
	flag.Parse()

	// Get the list of files and directories from command line arguments
	paths := flag.Args()

	if len(paths) == 0 {
		log.Fatal("Please provide a list of files to process.")
	}

	// Define the regular expressions to match == and != but not === or !==
	eqPattern := regexp.MustCompile(`(^|[^!=])==([^=]|$)`)
	nePattern := regexp.MustCompile(`(^|[^!])!=([^=]|$)`)

	// Process each provided path
	for _, path := range paths {
		processFile(path, eqPattern, nePattern, *writeFlag)
	}
}

func processFile(path string, eqPattern, nePattern *regexp.Regexp, writeBack bool) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatalf("Error opening file %s: %v\n", path, err)
	}
	defer file.Close()

	var modifiedLines []string
	var containsEqualityOps bool

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		modifiedLine := line

		// Check and replace == and !=
		if eqPattern.MatchString(line) || nePattern.MatchString(line) {
			containsEqualityOps = true
			fmt.Printf("%s: %s\n", path, line)
			modifiedLine = eqPattern.ReplaceAllString(modifiedLine, "$1===$2")
			modifiedLine = nePattern.ReplaceAllString(modifiedLine, "$1!==$2")
		}

		modifiedLines = append(modifiedLines, modifiedLine)
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading file %s: %v\n", path, err)
	}

	// Write back to file if the -w flag was provided
	if writeBack && containsEqualityOps {
		// Ensure the final output ends with a newline
		output := strings.Join(modifiedLines, "\n") + "\n"
		err := os.WriteFile(path, []byte(output), 0644)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(path)
	}
}
