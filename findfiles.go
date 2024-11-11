package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Usage: go run findfiles.go <regex>")
		os.Exit(1)
	}

	// Compile the regular expression from the command-line argument
	regexPattern := os.Args[1]
	regex, err := regexp.Compile(regexPattern)
	if err != nil {
		fmt.Printf("Invalid regular expression: %v\n", err)
		os.Exit(1)
	}

	// Get the current directory
	currentDir, err := os.Getwd()
	if err != nil {
		fmt.Printf("Error getting current directory: %v\n", err)
		os.Exit(1)
	}

	// Walk through the directory tree
	err = filepath.Walk(currentDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Check if the absolute path matches the regular expression
		if regex.MatchString(path) {
			absPath, err := filepath.Abs(path)
			if err != nil {
				return err
			}
			fmt.Println(absPath)
		}

		return nil
	})

	if err != nil {
		fmt.Printf("Error walking through directory: %v\n", err)
		os.Exit(1)
	}
}
