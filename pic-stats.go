package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"

	"github.com/rwcarlsen/goexif/exif"
	"github.com/rwcarlsen/goexif/mknote"
)

func main() {
	// Register EXIF support for maker notes (camera-specific information).
	exif.RegisterParsers(mknote.All...)

	// Start walking the directory tree from the current directory.
	err := filepath.Walk(".", processFile)
	if err != nil {
		fmt.Printf("Error walking through files: %v\n", err)
	}
}

func processFile(path string, info os.FileInfo, err error) error {
	if err != nil {
		fmt.Printf("Error processing file: %v\n", err)
		return err
	}

	if !info.IsDir() && isImageFile(path) {
		file, err := os.Open(path)
		if err != nil {
			fmt.Printf("Error opening file %s: %v\n", path, err)
			return nil
		}
		defer file.Close()

		// Get image dimensions.
		img, _, err := image.DecodeConfig(file)
		if err != nil {
			fmt.Printf("Error decoding image %s: %v\n", path, err)
			return nil
		}

		// Print image width and height.
		fmt.Printf("File: %s\n", path)
		fmt.Printf("Dimensions: %dx%d\n", img.Width, img.Height)

		// Reset the file pointer to the beginning of the file.
		file.Seek(0, 0)

		// Extract and print EXIF metadata.
		exifData, err := exif.Decode(file)
		if err != nil {
			fmt.Printf("Error decoding EXIF data for %s: %v\n", path, err)
		} else {
			printExifData(exifData)
		}

		fmt.Println()
	}
	return nil
}

func isImageFile(path string) bool {
	// Check for common image file extensions.
	ext := filepath.Ext(path)
	switch ext {
	case ".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp":
		return true
	default:
		return false
	}
}

func printExifData(x *exif.Exif) {
	// Get and print the DateTime tag.
	if dateTime, err := x.DateTime(); err == nil {
		fmt.Printf("Date Taken: %s\n", dateTime)
	}

	// Get and print the GPS coordinates.
	if lat, long, err := x.LatLong(); err == nil {
		fmt.Printf("Location: (%f, %f)\n", lat, long)
	} else {
		fmt.Println("No GPS data found.")
	}

	// Iterate through all EXIF tags.
	x.Walk(exifWalker{})
}

type exifWalker struct{}

func (w exifWalker) Walk(name exif.FieldName, tag *exif.Tag) error {
	fmt.Printf("%s: %s\n", name, tag)
	return nil
}
