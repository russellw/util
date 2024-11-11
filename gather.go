package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// copyFile copies a file from src to dst. If dst already exists, it will be overwritten.
func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("could not open source file %s: %v", src, err)
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("could not create destination file %s: %v", dst, err)
	}
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	if err != nil {
		return fmt.Errorf("could not copy data from %s to %s: %v", src, dst, err)
	}

	return nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: gather <file_with_filenames>")
		os.Exit(1)
	}

	listFile := os.Args[1]
	file, err := os.Open(listFile)
	if err != nil {
		fmt.Printf("Could not open file %s: %v\n", listFile, err)
		os.Exit(1)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		sourceFile := scanner.Text()
		baseName := filepath.Base(sourceFile)
		destFile := filepath.Join(".", baseName)

		err := copyFile(sourceFile, destFile)
		if err != nil {
			fmt.Printf("Failed to copy %s: %v\n", sourceFile, err)
		} else {
			fmt.Printf("Copied %s to %s\n", sourceFile, destFile)
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Error reading file list: %v\n", err)
		os.Exit(1)
	}
}
