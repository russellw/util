package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
)

func main() {
	// Command-line flags
	headerFlag := flag.Bool("header", false, "Specify if the first line contains column names")
	flag.Parse()

	var file *os.File
	var err error

	if flag.NArg() > 0 {
		file, err = os.Open(flag.Arg(0))
		if err != nil {
			panic(err)
		}
		defer file.Close()
	} else {
		file = os.Stdin
	}

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	fmt.Println("<table>")

	if *headerFlag && len(rows) > 0 {
		fmt.Println("<tr>")
		for _, col := range rows[0] {
			fmt.Printf("<th>%s\n", col)
		}
		rows = rows[1:]
	}

	for _, row := range rows {
		fmt.Println("<tr>")
		for _, col := range row {
			fmt.Printf("<td>%s\n", col)
		}
	}

	fmt.Println("</table>")
}
