package main

import (
	"image"
	"image/color"
	"image/png"
	"os"
)

type Point struct {
	x, y int
}

func main() {
	if len(os.Args) != 3 {
		panic("Usage: greenscreen <input.png> <output.png>")
	}

	inputFile, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	defer inputFile.Close()

	srcImage, err := png.Decode(inputFile)
	if err != nil {
		panic(err)
	}

	bounds := srcImage.Bounds()
	dstImage := image.NewRGBA(bounds)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			dstImage.Set(x, y, srcImage.At(x, y))
		}
	}

	visited := make(map[Point]bool)
	queue := []Point{}

	// Enqueue all edge pixels
	for x := bounds.Min.X; x < bounds.Max.X; x++ {
		queue = append(queue, Point{x, bounds.Min.Y}, Point{x, bounds.Max.Y - 1})
	}
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		queue = append(queue, Point{bounds.Min.X, y}, Point{bounds.Max.X - 1, y})
	}

	for len(queue) > 0 {
		p := queue[0]
		queue = queue[1:]

		if visited[p] {
			continue
		}
		visited[p] = true

		if isGreen(srcImage.At(p.x, p.y)) {
			dstImage.Set(p.x, p.y, color.RGBA{0, 0, 0, 0})

			// Add neighbors to the queue
			neighbors := []Point{
				{p.x + 1, p.y}, {p.x - 1, p.y},
				{p.x, p.y + 1}, {p.x, p.y - 1},
			}
			for _, n := range neighbors {
				if n.x >= bounds.Min.X && n.x < bounds.Max.X && n.y >= bounds.Min.Y && n.y < bounds.Max.Y {
					queue = append(queue, n)
				}
			}
		}
	}

	outputFile, err := os.Create(os.Args[2])
	if err != nil {
		panic(err)
	}
	defer outputFile.Close()

	err = png.Encode(outputFile, dstImage)
	if err != nil {
		panic(err)
	}
}

func isGreen(c color.Color) bool {
	r, g, b, _ := c.RGBA()
	return g > r && g > b
}
