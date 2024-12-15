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

// sortRules sorts rules and their properties alphabetically.
func sortRules(rules []Rule) {
	for i := range rules {
		sort.Strings(rules[i].selectors)
		sort.Strings(rules[i].properties)
		sortRules(rules[i].rules)
	}
	sort.Slice(rules, func(i, j int) bool {
		return strings.Join(rules[i].selectors, ", ") < strings.Join(rules[j].selectors, ", ")
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

	filename := "styles.css"
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
