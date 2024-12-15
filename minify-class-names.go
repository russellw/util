package main

import (
	"flag"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
)

func generateIdentifiers() []string {
	ids := []string{}
	for i := 'a'; i <= 'z'; i++ {
		ids = append(ids, string(i))
	}
	return ids
}

func main() {
	// Command-line flags
	write := flag.Bool("w", false, "overwrite the input file")
	flag.Parse()

	if len(flag.Args()) != 1 {
		fmt.Println("Usage: go run main.go [-w] <file.html>")
		os.Exit(1)
	}

	filePath := flag.Args()[0]

	// Read the input file
	data, err := os.ReadFile(filePath)
	if err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		os.Exit(1)
	}

	html := string(data)

	// Extract the <style> section
	styleRegex := regexp.MustCompile(`(?s)<style.*?>(.*?)</style>`)
	styleMatch := styleRegex.FindStringSubmatch(html)
	if styleMatch == nil {
		fmt.Println("No <style> section found in the HTML.")
		os.Exit(0)
	}

	// Process the style content to find class names
	styleContent := styleMatch[1]
	classRegex := regexp.MustCompile(`\.(\w[\w-]*)\s*{`)
	classMatches := classRegex.FindAllStringSubmatch(styleContent, -1)

	// Extract unique class names
	classNames := make(map[string]bool)
	for _, match := range classMatches {
		classNames[match[1]] = true
	}

	uniqueClassNames := make([]string, 0, len(classNames))
	for className := range classNames {
		uniqueClassNames = append(uniqueClassNames, className)
	}
	sort.Strings(uniqueClassNames)

	// Map class names to short identifiers
	identifiers := generateIdentifiers()
	if len(uniqueClassNames) > len(identifiers) {
		fmt.Println("Error: Too many class names to replace with single-letter identifiers.")
		os.Exit(1)
	}

	classMap := make(map[string]string)
	for i, className := range uniqueClassNames {
		classMap[className] = identifiers[i]
	}

	// Replace class names in the <style> section
	for className, shortName := range classMap {
		styleContent = regexp.MustCompile(`\.`+regexp.QuoteMeta(className)+`\b`).ReplaceAllString(styleContent, "."+shortName)
	}

	// Replace the modified <style> section back into the HTML
	html = strings.Replace(html, styleMatch[1], styleContent, 1)

	// Replace class names in `class` attributes in the HTML
	for className, shortName := range classMap {
		classAttrPattern := regexp.MustCompile(`class="([^"]*?)\b` + regexp.QuoteMeta(className) + `\b([^"]*?)"`)
		html = classAttrPattern.ReplaceAllString(html, `class="$1`+shortName+`$2"`)
	}

	// Write the output
	if *write {
		err = os.WriteFile(filePath, []byte(html), 0644)
		if err != nil {
			fmt.Printf("Error writing file: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("File successfully overwritten.")
	} else {
		fmt.Println(html)
	}
}
