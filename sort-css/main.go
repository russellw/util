package main

import (
	"flag"
	"fmt"
	"golang.org/x/net/html"
	"io"
	"io/ioutil"
	"os"
	"regexp"
	"sort"
	"strings"
)

// SortCSSProperties sorts the CSS properties alphabetically within a style block.
func SortCSSProperties(css string) string {
	// Regex to match CSS rules (selectors and their properties)
	ruleRegex := regexp.MustCompile(`(?ms)([^{]+){([^}]+)}`)
	return ruleRegex.ReplaceAllStringFunc(css, func(rule string) string {
		matches := ruleRegex.FindStringSubmatch(rule)
		if len(matches) != 3 {
			return rule // Return rule unchanged if it doesn't match
		}
		selector := strings.TrimSpace(matches[1])
		propertiesBlock := strings.TrimSpace(matches[2])

		// Split properties, sort, and join them back
		properties := strings.Split(propertiesBlock, ";")
		var sortedProps []string
		for _, prop := range properties {
			prop = strings.TrimSpace(prop)
			if prop != "" {
				sortedProps = append(sortedProps, prop)
			}
		}
		sort.Strings(sortedProps)

		return fmt.Sprintf("%s {%s;}", selector, strings.Join(sortedProps, ";"))
	})
}

// ProcessHTML reads the input HTML, sorts CSS properties inside <style> tags, and returns the updated HTML content.
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
		case html.DoctypeToken:
			token := tokenizer.Token()
			result.WriteString(fmt.Sprintf("<!doctype %s>", token.Data))
		case html.SelfClosingTagToken:
			token := tokenizer.Token()
			result.WriteString(renderToken(token))
		case html.TextToken:
			result.WriteString(string(tokenizer.Text()))
		case html.EndTagToken:
			token := tokenizer.Token()
			result.WriteString(renderToken(token))
		case html.StartTagToken:
			token := tokenizer.Token()
			if token.Data == "style" {
				// Handle <style> content separately
				styleContent := extractStyleContent(tokenizer)
				// Sort CSS properties within the style block
				sortedCSS := SortCSSProperties(styleContent)
				result.WriteString("<style>\n")
				result.WriteString(sortedCSS)
				result.WriteString("</style>")
			} else {
				result.WriteString(renderToken(token))
			}
		}
	}
}

// extractStyleContent extracts the content of the <style> block.
func extractStyleContent(tokenizer *html.Tokenizer) string {
	var sb strings.Builder
	for {
		tt := tokenizer.Next()
		switch tt {
		case html.TextToken:
			sb.WriteString(string(tokenizer.Text()))
		case html.EndTagToken:
			token := tokenizer.Token()
			if token.Data == "style" {
				return sb.String()
			}
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
		fmt.Println("Usage: go run sort-css.go [-w] <file1> [<file2> ...]")
		return
	}

	for _, filename := range flag.Args() {
		if err := processFile(filename, *writeBack); err != nil {
			fmt.Fprintf(os.Stderr, "Error processing %s: %v\n", filename, err)
		}
	}
}
