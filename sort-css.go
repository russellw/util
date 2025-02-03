package main

import (
	"bufio"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"sort"
	"strings"
)

// Rule represents a CSS rule.
type Rule struct {
	selectors  []string
	properties []string
	rules      []Rule
}

var lines []string

func readLines(path string) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		s := scanner.Text()
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		lines = append(lines, s)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}

func parseRule(i *int) Rule {
	var selectors []string
	for strings.HasSuffix(lines[*i], ",") {
		s := lines[*i]
		*i++
		s = strings.TrimSuffix(s, ",")
		s = strings.TrimSpace(s)
		selectors = append(selectors, s)
	}
	if !strings.HasSuffix(lines[*i], "{") {
		log.Fatal("syntax error")
	}
	s := lines[*i]
	*i++
	s = strings.TrimSuffix(s, "{")
	s = strings.TrimSpace(s)
	selectors = append(selectors, s)
	rule := Rule{selectors: selectors}
	for *i < len(lines) {
		if lines[*i] == "}" {
			*i++
			break
		}
		if strings.HasSuffix(lines[*i], "{") || strings.HasSuffix(lines[*i], ",") {
			rule.rules = append(rule.rules, parseRule(i))
			continue
		}
		s := lines[*i]
		*i++
		rule.properties = append(rule.properties, s)
	}
	return rule
}

// customLess compares two strings with custom rules:
// - Letters ('a'-'z', 'A'-'Z') come before '.' and '@'.
// - '.' comes after letters but before '@'.
// - '@' comes last.
func customLess(a, b string) bool {
	for i := 0; i < len(a) && i < len(b); i++ {
		ra, rb := customRank(a[i]), customRank(b[i])
		if ra != rb {
			return ra < rb
		}
	}
	return len(a) < len(b)
}

// customRank assigns a rank to a character based on custom rules.
func customRank(c byte) int {
	switch {
	case c >= 'a' && c <= 'z': // Lowercase letters
		return int(c)
	case c >= 'A' && c <= 'Z': // Uppercase letters
		return int(c)
	case c == '.': // '.' comes after letters
		return 256
	case c == '#':
		return 257
	case c == '@': // '@' comes after '.'
		return 258
	default: // Other characters
		return 300 + int(c)
	}
}

// sortRules sorts rules and their properties alphabetically.
func sortRules(rules []Rule) {
	for i := range rules {
		selectors := rules[i].selectors
		sort.Slice(selectors, func(i, j int) bool {
			return customLess(selectors[i], selectors[j])
		})
		sort.Strings(rules[i].properties)
		sortRules(rules[i].rules)
	}
	sort.Slice(rules, func(i, j int) bool {
		return customLess(strings.Join(rules[i].selectors, ", "), strings.Join(rules[j].selectors, ", "))
	})
}

// stringifyRules converts sorted rules back to a CSS string.
func stringifyRules(rules []Rule, indent string) string {
	var builder strings.Builder

	for _, rule := range rules {
		builder.WriteString(indent + strings.Join(rule.selectors, ", ") + " {\n")
		for _, prop := range rule.properties {
			builder.WriteString(indent + "  " + prop + "\n")
		}
		builder.WriteString(stringifyRules(rule.rules, indent+"  "))
		builder.WriteString(indent + "}\n")
	}

	return builder.String()
}

func main() {
	// Command-line flags.
	writeFlag := flag.Bool("w", false, "Overwrite the input file instead of printing")
	flag.Parse()

	filename := "style.css"
	if flag.NArg() > 0 {
		filename = flag.Arg(0)
	}

	// Read the input file.
	readLines(filename)

	// Parse, sort, and stringify CSS.
	var rules []Rule
	i := 0
	for i < len(lines) {
		rules = append(rules, parseRule(&i))
	}
	sortRules(rules)
	output := stringifyRules(rules, "")

	if *writeFlag {
		// Write back to the file.
		if err := ioutil.WriteFile(filename, []byte(output), 0644); err != nil {
			panic(fmt.Sprintf("Failed to write file %s: %v", filename, err))
		}
	} else {
		// Print to standard output.
		fmt.Println(output)
	}
}
