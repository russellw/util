package main

import (
	"flag"
	"fmt"
	"golang.org/x/net/html"
	"io"
	"io/ioutil"
	"os"
	"sort"
	"strings"
)

// SortAttributes sorts HTML tag attributes alphabetically by key.
func SortAttributes(token html.Token) html.Token {
	if len(token.Attr) > 1 {
		sort.Slice(token.Attr, func(i, j int) bool {
			return token.Attr[i].Key < token.Attr[j].Key
		})
	}
	return token
}

// ProcessHTML reads the input HTML, sorts attributes, and returns the updated HTML content.
func ProcessHTML(r io.Reader) (string, error) {
	var result strings.Builder
	tokenizer := html.NewTokenizer(r)

	for {
		tt := tokenizer.Next()
		switch tt {
		case html.ErrorToken:
			if tokenizer.Err() == io.EOF {
				return result.String(), nil
			}
			return "", tokenizer.Err()
		case html.StartTagToken, html.SelfClosingTagToken:
			token := tokenizer.Token()
			token = SortAttributes(token)
			result.WriteString(renderToken(token))
		case html.TextToken:
			result.WriteString(string(tokenizer.Text()))
		case html.EndTagToken:
			token := tokenizer.Token()
			result.WriteString(renderToken(token))
		}
	}
}

// renderToken converts an HTML token back to string form.
func renderToken(token html.Token) string {
	var sb strings.Builder
	switch token.Type {
	case html.StartTagToken, html.SelfClosingTagToken:
		sb.WriteString("<")
		sb.WriteString(token.Data)
		for _, attr := range token.Attr {
			sb.WriteString(fmt.Sprintf(` %s="%s"`, attr.Key, attr.Val))
		}
		if token.Type == html.SelfClosingTagToken {
			sb.WriteString("/>")
		} else {
			sb.WriteString(">")
		}
	case html.EndTagToken:
		sb.WriteString(fmt.Sprintf("</%s>", token.Data))
	}
	return sb.String()
}

// processFile processes a single HTML file and writes the result either to stdout or back to the file.
func processFile(filename string, writeBack bool) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	processedHTML, err := ProcessHTML(file)
	if err != nil {
		return err
	}

	if writeBack {
		err = ioutil.WriteFile(filename, []byte(processedHTML), 0644)
		if err != nil {
			return err
		}
	} else {
		fmt.Println(processedHTML)
	}

	return nil
}

func main() {
	writeBack := flag.Bool("w", false, "write result back to input file")
	flag.Parse()

	if flag.NArg() == 0 {
		fmt.Println("Usage: go run sort-html-attrs.go [-w] <file1> [<file2> ...]")
		return
	}

	for _, filename := range flag.Args() {
		if err := processFile(filename, *writeBack); err != nil {
			fmt.Fprintf(os.Stderr, "Error processing %s: %v\n", filename, err)
		}
	}
}
