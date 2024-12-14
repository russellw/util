package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"sort"
	"strings"
)

// Rule represents a CSS rule.
type Rule struct {
	Selector    string
	Properties  []string
	NestedRules []Rule
}

// parseCSS recursively parses CSS into a slice of Rules.
func parseCSS(input string) []Rule {
	var rules []Rule
	lines := strings.Split(input, "\n")
	stack := []Rule{}

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if strings.HasSuffix(line, "{") {
			// Start of a new rule.
			selector := strings.TrimSpace(strings.TrimSuffix(line, "{"))
			stack = append(stack, Rule{Selector: selector})
		} else if line == "}" {
			// End of a rule.
			if len(stack) == 0 {
				panic("Unbalanced braces detected in CSS")
			}

			current := stack[len(stack)-1]
			stack = stack[:len(stack)-1]

			if len(stack) > 0 {
				stack[len(stack)-1].NestedRules = append(stack[len(stack)-1].NestedRules, current)
			} else {
				rules = append(rules, current)
			}
		} else {
			// Property or nested selector line.
			if len(stack) == 0 {
				panic("Property outside of any rule detected")
			}

			// Check if this line starts with a valid property or is part of a nested rule.
			if strings.Contains(line, ":") {
				stack[len(stack)-1].Properties = append(stack[len(stack)-1].Properties, line)
			} else {
				panic("Unexpected line in CSS: " + line)
			}
		}
	}

	if len(stack) != 0 {
		panic("Unbalanced braces detected in CSS")
	}

	return rules
}

// sortRules sorts rules and their properties alphabetically.
func sortRules(rules []Rule) {
	sort.Slice(rules, func(i, j int) bool {
		return rules[i].Selector < rules[j].Selector
	})

	for i := range rules {
		sort.Strings(rules[i].Properties)
		sortRules(rules[i].NestedRules)
	}
}

// stringifyRules converts sorted rules back to a CSS string.
func stringifyRules(rules []Rule, indent string) string {
	var builder strings.Builder

	for _, rule := range rules {
		builder.WriteString(indent + rule.Selector + " {\n")
		for _, prop := range rule.Properties {
			builder.WriteString(indent + "  " + prop + "\n")
		}
		builder.WriteString(stringifyRules(rule.NestedRules, indent+"  "))
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
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(fmt.Sprintf("Failed to read file %s: %v", filename, err))
	}

	css := string(content)
	if strings.Contains(css, "/*") || strings.Contains(css, "*/") {
		panic("Comments are not supported in the input CSS")
	}

	// Parse, sort, and stringify CSS.
	rules := parseCSS(css)
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
