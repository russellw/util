package main

import (
	"reflect"
	"testing"
)

func TestTrimBlankLines(t *testing.T) {
	tests := []struct {
		name     string
		input    []string
		expected []string
	}{
		{
			name:     "No blank lines at end",
			input:    []string{"line1", "line2", "line3"},
			expected: []string{"line1", "line2", "line3"},
		},
		{
			name:     "One blank line at end",
			input:    []string{"line1", "line2", "line3", ""},
			expected: []string{"line1", "line2", "line3"},
		},
		{
			name:     "Multiple blank lines at end",
			input:    []string{"line1", "line2", "line3", "", "  ", "\t"},
			expected: []string{"line1", "line2", "line3"},
		},
		{
			name:     "Blank lines only",
			input:    []string{"", " ", "\t", ""},
			expected: []string{},
		},
		{
			name:     "Mixed with blank lines in the middle",
			input:    []string{"line1", "line2", "", "line3", "  ", ""},
			expected: []string{"line1", "line2", "", "line3"},
		},
		{
			name:     "Empty input",
			input:    []string{},
			expected: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := trimBlankLines(tt.input)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("got %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestIsComment checks various cases for the isComment function
func TestIsComment(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{"// This is a comment", true},
		{"  // Indented comment", true},
		{"This is not a comment", false},
		{"//", true},
		{"  //", true},
		{" //Partial comment", true},
		{"/* Block comment */", false},
		{"  /* Not a line comment */", false},
	}

	for _, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			result := isComment(test.input)
			if result != test.expected {
				t.Errorf("For input '%s', expected %v but got %v", test.input, test.expected, result)
			}
		})
	}
}

func TestParseChunks_NoSpecialChunks(t *testing.T) {
	isComment := func(s string) bool { return false }
	beginSpecial := func(s string) string { return "" }
	endSpecial := func(s string) EndSpecialKind { return endSpecialNo }

	lines := []string{
		"line 1",
		"line 2",
		"line 3",
	}

	expected := []Chunk{
		{name: "", lines: []string{"line 1", "line 2", "line 3"}},
	}

	result := parseChunks(isComment, beginSpecial, endSpecial, lines)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
}

func TestParseChunks_WithSpecialChunks(t *testing.T) {
	isComment := func(s string) bool { return false }
	beginSpecial := func(s string) string {
		if s == "special start" {
			return "special"
		}
		return ""
	}
	endSpecial := func(s string) EndSpecialKind {
		if s == "special end" {
			return endSpecialInclude
		}
		return endSpecialNo
	}

	lines := []string{
		"line 1",
		"special start",
		"special content",
		"special end",
		"line 2",
	}

	expected := []Chunk{
		{name: "", lines: []string{"line 1"}},
		{name: "special", lines: []string{"special start", "special content", "special end"}},
		{name: "", lines: []string{"line 2"}},
	}

	result := parseChunks(isComment, beginSpecial, endSpecial, lines)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
}

func TestParseChunks_ExcludeEndSpecial(t *testing.T) {
	isComment := func(s string) bool { return false }
	beginSpecial := func(s string) string {
		if s == "special start" {
			return "special"
		}
		return ""
	}
	endSpecial := func(s string) EndSpecialKind {
		if s == "special end" {
			return endSpecialExclude
		}
		return endSpecialNo
	}

	lines := []string{
		"line 1",
		"special start",
		"special content",
		"special end",
		"line 2",
	}

	expected := []Chunk{
		{name: "", lines: []string{"line 1"}},
		{name: "special", lines: []string{"special start", "special content"}},
		{name: "", lines: []string{"special end", "line 2"}},
	}

	result := parseChunks(isComment, beginSpecial, endSpecial, lines)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
}

func TestParseChunks_WithComments(t *testing.T) {
	isComment := func(s string) bool {
		return s == "# comment"
	}
	beginSpecial := func(s string) string { return "" }
	endSpecial := func(s string) EndSpecialKind { return endSpecialNo }

	lines := []string{
		"line 1",
		"# comment",
		"line 2",
		"line 3",
	}

	expected := []Chunk{
		{name: "", lines: []string{"line 1", "# comment", "line 2", "line 3"}},
	}

	result := parseChunks(isComment, beginSpecial, endSpecial, lines)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
}

func TestParseChunks_EmptyLines(t *testing.T) {
	isComment := func(s string) bool { return false }
	beginSpecial := func(s string) string { return "" }
	endSpecial := func(s string) EndSpecialKind { return endSpecialNo }

	lines := []string{}

	result := parseChunks(isComment, beginSpecial, endSpecial, lines)
	if result != nil && len(result) != 0 {
		t.Errorf("Expected empty or nil slice, but got %v", result)
	}
}
