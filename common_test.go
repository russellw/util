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

// Helper function to compare slices of Range, accounting for nil vs empty slice differences
func areEqualRanges(a, b []Range) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

// Test case 1: Empty slice
func TestGetRanges_EmptySlice(t *testing.T) {
	f := func(x float64) bool { return x > 0 }
	v := []float64{}

	expected := []Range{}
	result := getRanges(f, v)

	if !areEqualRanges(result, expected) {
		t.Errorf("TestGetRanges_EmptySlice failed: expected %v, got %v", expected, result)
	}
}

// Test case 2: All values matching predicate
func TestGetRanges_AllValuesMatch(t *testing.T) {
	f := func(x float64) bool { return x > 0 }
	v := []float64{1.0, 2.5, 3.1}

	expected := []Range{{i: 0, j: 3}}
	result := getRanges(f, v)

	if !areEqualRanges(result, expected) {
		t.Errorf("TestGetRanges_AllValuesMatch failed: expected %v, got %v", expected, result)
	}
}

// Test case 3: No values matching predicate
func TestGetRanges_NoValuesMatch(t *testing.T) {
	f := func(x float64) bool { return x > 0 }
	v := []float64{-1.0, -2.5, -3.1}

	expected := []Range{}
	result := getRanges(f, v)

	if !areEqualRanges(result, expected) {
		t.Errorf("TestGetRanges_NoValuesMatch failed: expected %v, got %v", expected, result)
	}
}

// Test case 4: Mixed values with one positive range
func TestGetRanges_MixedValuesOneRange(t *testing.T) {
	f := func(x float64) bool { return x > 0 }
	v := []float64{-1.0, 2.5, 3.1, -3.1}

	expected := []Range{{i: 1, j: 3}}
	result := getRanges(f, v)

	if !areEqualRanges(result, expected) {
		t.Errorf("TestGetRanges_MixedValuesOneRange failed: expected %v, got %v", expected, result)
	}
}

// Test case 5: Mixed values with two positive ranges
func TestGetRanges_MixedValuesTwoRanges(t *testing.T) {
	f := func(x float64) bool { return x > 0 }
	v := []float64{-1.0, 2.5, 3.1, -3.1, 1.2}

	expected := []Range{{i: 1, j: 3}, {i: 4, j: 5}}
	result := getRanges(f, v)

	if !areEqualRanges(result, expected) {
		t.Errorf("TestGetRanges_MixedValuesTwoRanges failed: expected %v, got %v", expected, result)
	}
}

// Test case 6: Single matching value
func TestGetRanges_SingleMatchingValue(t *testing.T) {
	f := func(x float64) bool { return x > 0 }
	v := []float64{-1.0, 2.5, -3.1}

	expected := []Range{{i: 1, j: 2}}
	result := getRanges(f, v)

	if !areEqualRanges(result, expected) {
		t.Errorf("TestGetRanges_SingleMatchingValue failed: expected %v, got %v", expected, result)
	}
}
