package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"github.com/Kodeworks/golang-image-ico"
	"golang.org/x/image/draw"
)

func main() {
	// Parse the input file as a positional argument
	flag.Parse()
	if flag.NArg() < 1 {
		fmt.Println("Usage: png-to-ico <input.png>")
		return
	}
	inputFile := flag.Arg(0)

	// Set the output file name by changing the extension to .ico
	outputFile := strings.TrimSuffix(inputFile, filepath.Ext(inputFile)) + ".ico"

	// Open the PNG file
	file, err := os.Open(inputFile)
	if err != nil {
		fmt.Println("Error opening input file:", err)
		return
	}
	defer file.Close()

	// Decode the PNG image
	img, err := png.Decode(file)
	if err != nil {
		fmt.Println("Error decoding PNG file:", err)
		return
	}

	// Define icon sizes (for Windows compatibility)
	sizes := []int{16, 32, 48, 64, 128, 256}

	// Create an ICO file
	icoFile, err := os.Create(outputFile)
	if err != nil {
		fmt.Println("Error creating ICO file:", err)
		return
	}
	defer icoFile.Close()

	// Add each resized image to the ICO file
	for _, size := range sizes {
		resizedImg := resizeImage(img, size, size)
		if err := ico.Encode(icoFile, resizedImg); err != nil {
			fmt.Println("Error adding resized image to ICO:", err)
			return
		}
	}

	fmt.Println("ICO file created successfully:", outputFile)
}

// resizeImage resizes the source image to the specified width and height
func resizeImage(src image.Image, width, height int) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Over, nil)
	return dst
}
