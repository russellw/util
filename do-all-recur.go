package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

// Function to execute a program on a file
func runProgramOnFile(program string, args []string, filePath string) {
	// Append the file path to the args
	allArgs := append(args, filePath)

	// Execute the command
	cmd := exec.Command(program, allArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()

	if err != nil {
		fmt.Println(filePath)
		if exitError, ok := err.(*exec.ExitError); ok {
			log.Fatalf("program returned non-zero exit code: %v", exitError.ExitCode())
		}
		log.Fatalf("failed to execute program: %v", err)
	}
}

// Function to recursively traverse the directory and run the program on each file
func walkDirAndRunProgram(root string, program string, args []string) {
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Fatalf("error accessing file path: %v", err)
		}

		// If it's a file, run the program on it
		if !info.IsDir() {
			runProgramOnFile(program, args, path)
		}

		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	// Check if at least two arguments (program and one arg) are provided
	if len(os.Args) < 2 {
		log.Fatalf("Usage: %s program arg1 arg2... \n", os.Args[0])
	}

	// Program and its arguments
	program := os.Args[1]
	args := os.Args[2:]

	// Current directory
	currentDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get current directory: %v", err)
	}

	// Traverse the directory tree and run the program on each file
	walkDirAndRunProgram(currentDir, program, args)
}
