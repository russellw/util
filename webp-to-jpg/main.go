package main

import (
	"fmt"
	"image/jpeg"
	"log"
	"os"
	"path/filepath"

	"golang.org/x/image/webp"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Usage: go run main.go <input.webp>")
		os.Exit(1)
	}

	inputFile := os.Args[1]

	// Open the input .webp file
	file, err := os.Open(inputFile)
	if err != nil {
		log.Fatalf("Failed to open input file: %v", err)
	}
	defer file.Close()

	// Decode the .webp image
	img, err := webp.Decode(file)
	if err != nil {
		log.Fatalf("Failed to decode webp image: %v", err)
	}

	// Change the file extension to .jpg
	outputFile := changeExtension(inputFile, ".jpg")

	// Create the output .jpg file
	outFile, err := os.Create(outputFile)
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}
	defer outFile.Close()

	// Encode the image to .jpg format
	options := &jpeg.Options{Quality: 90} // Set the quality for the JPEG
	err = jpeg.Encode(outFile, img, options)
	if err != nil {
		log.Fatalf("Failed to encode JPEG: %v", err)
	}

	fmt.Printf("Conversion complete: %s\n", outputFile)
}

// changeExtension changes the file extension of a given filename
func changeExtension(filename, newExt string) string {
	return filepath.Base(filename[:len(filename)-len(filepath.Ext(filename))]) + newExt
}
