package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	// Ensure at least three arguments: in-place, file, and filter-program
	if len(os.Args) < 3 {
		fmt.Println("Usage: in-place <file> <filter-program> [args...]")
		os.Exit(1)
	}

	// Get the file and filter-program from the arguments
	filePath := os.Args[1]
	filterProgram := os.Args[2]
	filterArgs := os.Args[3:]

	// Read the original file
	originalContent, err := ioutil.ReadFile(filePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		os.Exit(1)
	}

	// Prepare the command to run the filter program
	cmd := exec.Command(filterProgram, filterArgs...)
	cmd.Stdin = bytes.NewReader(originalContent)

	// Capture the output from the filter program
	var outputBuffer bytes.Buffer
	cmd.Stdout = &outputBuffer

	// Run the filter program
	err = cmd.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Filter program failed: %v\n", err)
		os.Exit(1)
	}

	// Check if the output is the same as the original content
	filteredContent := outputBuffer.Bytes()
	if bytes.Equal(originalContent, filteredContent) {
		// If identical, exit quietly
		os.Exit(0)
	}

	// If content is different, move the original file to a temporary backup
	tempDir := os.TempDir()
	backupFile := filepath.Join(tempDir, filepath.Base(filePath))

	err = os.Rename(filePath, backupFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating backup: %v\n", err)
		os.Exit(1)
	}

	// Write the new filtered content to the original file
	err = ioutil.WriteFile(filePath, filteredContent, 0644)
	if err != nil {
		// Attempt to restore the backup in case of error
		_ = os.Rename(backupFile, filePath)
		fmt.Fprintf(os.Stderr, "Error writing new content: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("%s\n", filePath)
}
