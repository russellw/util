package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/nfnt/resize"
)

// IconDir structure defines the header of the ICO file
type IconDir struct {
	Reserved   uint16 // Reserved (must be 0)
	Type       uint16 // Resource Type (1 for icons)
	ImageCount uint16 // Number of images in the file
}

// IconDirEntry structure defines each icon entry in the ICO file
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

// Resize image to specified dimensions
func resizeImage(img image.Image, width, height int) image.Image {
	return resize.Resize(uint(width), uint(height), img, resize.Lanczos3)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: png-to-ico <input.png>")
		return
	}
	inputFile := os.Args[1]
	outputFile := inputFile[:len(inputFile)-4] + ".ico"

	// Open PNG file
	file, err := os.Open(inputFile)
	if err != nil {
		fmt.Println("Error opening PNG file:", err)
		return
	}
	defer file.Close()

	// Decode PNG
	img, err := png.Decode(file)
	if err != nil {
		fmt.Println("Error decoding PNG file:", err)
		return
	}

	// Define icon sizes (for Windows compatibility)
	//sizes := []int{16, 32, 48, 64, 128, 256}
	sizes := []int{16, 32, 48}
	entries := make([]IconDirEntry, len(sizes))
	imageData := make([][]byte, len(sizes))

	// Resize images and encode them to PNG format in memory
	for i, size := range sizes {
		resizedImg := resizeImage(img, size, size)

		var imgBuf bytes.Buffer
		if err := png.Encode(&imgBuf, resizedImg); err != nil {
			fmt.Println("Error encoding resized image to PNG:", err)
			return
		}

		buf := imgBuf.Bytes()
		imageData[i] = buf
		entries[i] = IconDirEntry{
			Width:       uint8(size % 256), // 256 translates to 0 in ICO (indicating 256)
			Height:      uint8(size % 256),
			ColorCount:  0,
			Reserved:    0,
			Planes:      1,
			BitCount:    32,
			SizeInBytes: uint32(len(buf)),
			ImageOffset: 0, // Placeholder, will set later
		}
	}

	// Create ICO file
	icoFile, err := os.Create(outputFile)
	if err != nil {
		fmt.Println("Error creating ICO file:", err)
		return
	}
	defer icoFile.Close()

	// Write IconDir header
	header := IconDir{
		Reserved:   0,
		Type:       1, // Icon type
		ImageCount: uint16(len(entries)),
	}
	if err := binary.Write(icoFile, binary.LittleEndian, header); err != nil {
		fmt.Println("Error writing ICO header:", err)
		return
	}

	// Calculate offsets and write entries
	offset := 6 + 16*len(entries) // Initial offset after header and entries
	for i := range entries {
		entries[i].ImageOffset = uint32(offset)
		offset += int(entries[i].SizeInBytes)
	}

	// Write each entry
	for _, entry := range entries {
		if err := binary.Write(icoFile, binary.LittleEndian, entry); err != nil {
			fmt.Println("Error writing ICO entry:", err)
			return
		}
	}

	// Write image data
	for _, data := range imageData {
		if _, err := icoFile.Write(data); err != nil {
			fmt.Println("Error writing image data to ICO:", err)
			return
		}
	}

	fmt.Println("ICO file created successfully:", outputFile)
}
