package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// FileInfo stores the information of a file
type FileInfo struct {
	Path string
	Size int64
	ModTime time.Time
}

// WalkDir recursively walks through a directory and stores the file info
func WalkDir(dir string) (map[string]FileInfo, error) {
	files := make(map[string]FileInfo)
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			relPath, err := filepath.Rel(dir, path)
			if err != nil {
				return err
			}
			files[relPath] = FileInfo{
				Path:    path,
				Size:    info.Size(),
				ModTime: info.ModTime(),
			}
		}
		return nil
	})
	return files, err
}

// CompareDirectories compares two directories and prints differences
func CompareDirectories(dir1, dir2 string) error {
	files1, err := WalkDir(dir1)
	if err != nil {
		return fmt.Errorf("error walking directory %s: %v", dir1, err)
	}

	files2, err := WalkDir(dir2)
	if err != nil {
		return fmt.Errorf("error walking directory %s: %v", dir2, err)
	}

	// Check for files in dir1 but not in dir2 or with different properties
	for relPath, info1 := range files1 {
		info2, exists := files2[relPath]
		if !exists {
			fmt.Printf("File %s exists in %s but not in %s\n", relPath, dir1, dir2)
		} else if info1.Size != info2.Size || !info1.ModTime.Equal(info2.ModTime) {
			fmt.Printf("File %s differs:\n", relPath)
			fmt.Printf("  %s - Size: %d, ModTime: %v\n", dir1, info1.Size, info1.ModTime)
			fmt.Printf("  %s - Size: %d, ModTime: %v\n", dir2, info2.Size, info2.ModTime)
		}
	}

	// Check for files in dir2 but not in dir1
	for relPath := range files2 {
		if _, exists := files1[relPath]; !exists {
			fmt.Printf("File %s exists in %s but not in %s\n", relPath, dir2, dir1)
		}
	}

	return nil
}

func main() {
	dir1 := "path/to/directory1"
	dir2 := "path/to/directory2"

	err := CompareDirectories(dir1, dir2)
	if err != nil {
		fmt.Println("Error comparing directories:", err)
	}
}
