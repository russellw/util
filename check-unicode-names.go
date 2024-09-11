package main

import (
	"fmt"
	"os"
	"path/filepath"
	"unicode"
)

// checkDir recursively checks directories for files or subdirectories with non-ASCII names.
func checkDir(path string) error {
	err := filepath.Walk(path, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		// Check if the file or directory name contains non-ASCII characters
		if !isASCII(info.Name()) {
			fmt.Println(p)
		}
		return nil
	})
	return err
}

// isASCII checks if all characters in the string are ASCII.
func isASCII(s string) bool {
	for _, r := range s {
		if r > unicode.MaxASCII {
			return false
		}
	}
	return true
}

func main() {
	// Replace this with the directory path you want to start from
	rootDir := "."

	err := checkDir(rootDir)
	if err != nil {
		fmt.Println("Error:", err)
	}
}
