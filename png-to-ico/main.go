package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/Kodeworks/golang-image-ico"
	"golang.org/x/image/draw"
)

func main() {
	inputFile := flag.String("input", "input.png", "Path to input PNG file")
	outputFile := flag.String("output", "output.ico", "Path to output ICO file")
	flag.Parse()

	// Open the PNG file
	file, err := os.Open(*inputFile)
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
	icoFile, err := os.Create(*outputFile)
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

	fmt.Println("ICO file created successfully:", *outputFile)
}

// resizeImage resizes the source image to the specified width and height
func resizeImage(src image.Image, width, height int) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Over, nil)
	return dst
}
