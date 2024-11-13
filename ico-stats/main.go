package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"
)

type IconDir struct {
	Reserved   uint16 // Reserved (must be 0)
	Type       uint16 // Resource Type (1 for icons)
	ImageCount uint16 // Number of images in the file
}

type IconDirEntry struct {
	Width       uint8  // Width of the image
	Height      uint8  // Height of the image
	ColorCount  uint8  // Number of colors (0 if more than 256)
	Reserved    uint8  // Reserved (must be 0)
	Planes      uint16 // Color planes
	BitCount    uint16 // Bits per pixel
	SizeInBytes uint32 // Size of the image data in bytes
	ImageOffset uint32 // Offset of the image data from the start of the file
}

func main() {
	flag.Parse()
	if flag.NArg() < 1 {
		fmt.Println("Usage: ico-metadata <file.ico>")
		return
	}
	inputFile := flag.Arg(0)

	// Open the ICO file
	file, err := os.Open(inputFile)
	if err != nil {
		fmt.Println("Error opening ICO file:", err)
		return
	}
	defer file.Close()

	// Read the ICO header
	var header IconDir
	if err := binary.Read(file, binary.LittleEndian, &header); err != nil {
		fmt.Println("Error reading ICO header:", err)
		return
	}

	if header.Type != 1 {
		fmt.Println("Not a valid ICO file")
		return
	}

	fmt.Printf("ICO file: %s\n", inputFile)
	fmt.Printf("Number of images: %d\n\n", header.ImageCount)

	// Read each image entry
	for i := 0; i < int(header.ImageCount); i++ {
		var entry IconDirEntry
		if err := binary.Read(file, binary.LittleEndian, &entry); err != nil {
			fmt.Printf("Error reading entry %d: %v\n", i+1, err)
			return
		}

		// Handle 0 values for width or height by setting them to 256
		width := int(entry.Width)
		if width == 0 {
			width = 256
		}

		height := int(entry.Height)
		if height == 0 {
			height = 256
		}

		fmt.Printf("Image %d:\n", i+1)
		fmt.Printf("  Width: %d pixels\n", width)
		fmt.Printf("  Height: %d pixels\n", height)
		fmt.Printf("  Color Depth: %d bits\n", entry.BitCount)
		fmt.Printf("  Size in Bytes: %d\n", entry.SizeInBytes)
		fmt.Printf("  Offset: %d\n", entry.ImageOffset)
		fmt.Println()
	}
}
