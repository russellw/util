package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

// PNGHeader is the fixed 8-byte signature for PNG files.
var PNGHeader = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}

// Chunk represents a PNG chunk.
type Chunk struct {
	Length uint32
	Type   string
	Data   []byte
	CRC    uint32
}

func main() {
	// Get filenames from command-line arguments or glob for PNG files in the current directory.
	var filenames []string
	if len(os.Args) > 1 {
		filenames = expandGlobs(os.Args[1:])
	} else {
		var err error
		filenames, err = filepath.Glob("*.png")
		if err != nil {
			fmt.Println("Error finding PNG files:", err)
			return
		}
	}

	if len(filenames) == 0 {
		fmt.Println("No PNG files found.")
		return
	}

	// Inspect each file.
	for _, filename := range filenames {
		fmt.Printf("Inspecting file: %s\n", filename)
		if err := inspectPNG(filename); err != nil {
			fmt.Printf("Error inspecting %s: %v\n", filename, err)
		}
		fmt.Println(strings.Repeat("-", 40))
	}
}

// expandGlobs handles globbing for filenames, ensuring compatibility with Windows.
func expandGlobs(args []string) []string {
	var result []string
	for _, pattern := range args {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			fmt.Printf("Error expanding glob %s: %v\n", pattern, err)
			continue
		}
		result = append(result, matches...)
	}
	return result
}

// inspectPNG opens a PNG file, validates its signature, and extracts metadata.
func inspectPNG(filename string) error {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	if len(data) < len(PNGHeader) || !bytes.Equal(data[:len(PNGHeader)], PNGHeader) {
		return fmt.Errorf("file is not a valid PNG")
	}

	fmt.Println("Valid PNG signature detected.")

	chunks, err := parseChunks(data[len(PNGHeader):])
	if err != nil {
		return fmt.Errorf("failed to parse chunks: %w", err)
	}

	printChunks(chunks)
	return nil
}

// parseChunks reads chunks from PNG data.
func parseChunks(data []byte) ([]Chunk, error) {
	var chunks []Chunk
	for len(data) > 0 {
		if len(data) < 8 {
			return nil, fmt.Errorf("insufficient data for chunk")
		}

		// Read length and type.
		length := binary.BigEndian.Uint32(data[:4])
		chunkType := string(data[4:8])

		if len(data) < int(8+length+4) {
			return nil, fmt.Errorf("chunk data incomplete for chunk type: %s", chunkType)
		}

		// Extract data and CRC.
		chunkData := data[8 : 8+length]
		crc := binary.BigEndian.Uint32(data[8+length : 8+length+4])

		chunks = append(chunks, Chunk{
			Length: length,
			Type:   chunkType,
			Data:   chunkData,
			CRC:    crc,
		})

		// Move to the next chunk.
		data = data[8+length+4:]
	}
	return chunks, nil
}

// printChunks prints information about PNG chunks, including metadata.
func printChunks(chunks []Chunk) {
	for _, chunk := range chunks {
		fmt.Printf("Chunk Type: %s, Length: %d bytes\n", chunk.Type, chunk.Length)
		if chunk.Type == "tEXt" || chunk.Type == "zTXt" || chunk.Type == "iTXt" {
			// Decode text-based metadata.
			fmt.Printf("  Metadata: %s\n", string(chunk.Data))
		} else if chunk.Type == "IHDR" {
			// Print image header details.
			width := binary.BigEndian.Uint32(chunk.Data[:4])
			height := binary.BigEndian.Uint32(chunk.Data[4:8])
			fmt.Printf("  Image Dimensions: %dx%d\n", width, height)
		}
	}
}
