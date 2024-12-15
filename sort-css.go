package main

import (
"os"
	"flag"
	"fmt"
	"log"
	"bufio"
	"io/ioutil"
	"sort"
	"strings"
)

// Rule represents a CSS rule.
type Rule struct {
	Selectors   []string
	Properties  []string
	NestedRules []Rule
}

var  lines[]string

func readLines(path string) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines,strings.TrimSpace( scanner.Text()))
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}

func parseRule(i* int ) Rule {
var selectors[]string
		for strings.HasSuffix(lines[*i], ",") {
		s:=lines[*i]
			s = strings.TrimSuffix(s, "{")
			s = strings.TrimSpace(s)
			selectors=append(selectors,s)
			*i++
		}
		if !strings.HasSuffix(lines[*i], "{")  {
		log.Fatal("syntax error");
	}
	rule:=Rule{Selectors: selectors}
	return rule
}

// parseCSS recursively parses CSS into a slice of Rules.
func parseCSS(input string) []Rule {
	var rules []Rule
	stack := []Rule{}

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if strings.HasSuffix(line, "{") {
			// Start of a new rule.
			selector := strings.TrimSpace(strings.TrimSuffix(line, "{"))
			selectors := strings.Split(selector, ",")
			for i := range selectors {
				selectors[i] = strings.TrimSpace(selectors[i])
			}
			stack = append(stack, Rule{Selectors: selectors})
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
			// Property line.
			if len(stack) == 0 {
				panic("Property outside of any rule detected")
			}

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
		return strings.Join(rules[i].Selectors, ", ") < strings.Join(rules[j].Selectors, ", ")
	})

	for i := range rules {
		sort.Strings(rules[i].Selectors)
		sort.Strings(rules[i].Properties)
		sortRules(rules[i].NestedRules)
	}
}

// stringifyRules converts sorted rules back to a CSS string.
func stringifyRules(rules []Rule, indent string) string {
	var builder strings.Builder

	for _, rule := range rules {
		builder.WriteString(indent + strings.Join(rule.Selectors, ", ") + " {\n")
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
	readLines(filename)

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
