package main

import (
	"reflect"
	"regexp"
	"testing"
)

// Unit test for appendUnique function
func TestAppendUnique(t *testing.T) {
	tests := []struct {
		name     string
		v        []string
		s        string
		expected []string
	}{
		{"Add unique string", []string{"apple", "banana"}, "cherry", []string{"apple", "banana", "cherry"}},
		{"Duplicate string", []string{"apple", "banana"}, "apple", []string{"apple", "banana"}},
		{"Empty slice", []string{}, "apple", []string{"apple"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := appendUnique(tt.v, tt.s)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Unit test for contains function
func TestContains(t *testing.T) {
	tests := []struct {
		name     string
		slice    []string
		str      string
		expected bool
	}{
		{"Contains string", []string{"apple", "banana", "cherry"}, "banana", true},
		{"Does not contain string", []string{"apple", "banana", "cherry"}, "grape", false},
		{"Empty slice", []string{}, "apple", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := contains(tt.slice, tt.str)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestCopyEmptySlice(t *testing.T) {
	// Test Case 2: Copy an empty slice
	input := []string{}
	expected := []string{}

	result := copySlice(input)

	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
}

func TestCopyNilSlice(t *testing.T) {
	// Test Case 4: Copy a nil slice
	var input []string
	var expected []string

	result := copySlice(input)

	if !eqStrings(result, expected) {
		t.Errorf("Expected nil slice, but got %v", result)
	}
}

func TestCopySingleElementSlice(t *testing.T) {
	// Test Case 3: Copy a slice with one element
	input := []string{"single"}
	expected := []string{"single"}

	result := copySlice(input)

	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
}

func TestCopySlice(t *testing.T) {
	// Test Case 1: Copy a slice of strings
	input := []string{"apple", "banana", "cherry"}
	expected := []string{"apple", "banana", "cherry"}

	result := copySlice(input)

	// Verify if the result matches the expected output
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, but got %v", expected, result)
	}

	// Ensure the original slice is not affected by changes to the copy
	result[0] = "modified"
	if reflect.DeepEqual(result, input) {
		t.Errorf("Original slice was modified. Expected original: %v, got: %v", input, result)
	}
}

func TestFindString_Ef(t *testing.T) {
	re := regexp.MustCompile(`ef`)
	v := []string{"abc", "test1", "def", "test2"}
	index := findString(re, v, 0)
	if index != 2 {
		t.Errorf("Expected index 2, got %d", index)
	}
}

func TestFindString_EmptySlice(t *testing.T) {
	re := regexp.MustCompile(`^test`)
	v := []string{}
	index := findString(re, v, 0)
	if index != -1 {
		t.Errorf("Expected index -1, got %d", index)
	}
}

func TestFindString_Found(t *testing.T) {
	re := regexp.MustCompile(`^test`)
	v := []string{"abc", "test1", "def", "test2"}
	index := findString(re, v, 0)
	if index != 1 {
		t.Errorf("Expected index 1, got %d", index)
	}
}

func TestFindString_InvalidStartIndex(t *testing.T) {
	re := regexp.MustCompile(`^test`)
	v := []string{"abc", "test1", "def", "test2"}
	index := findString(re, v, len(v)+1) // Start index out of range
	if index != -1 {
		t.Errorf("Expected index -1, got %d", index)
	}
}

func TestFindString_NotFound(t *testing.T) {
	re := regexp.MustCompile(`^test`)
	v := []string{"abc", "def", "ghi"}
	index := findString(re, v, 0)
	if index != -1 {
		t.Errorf("Expected index -1, got %d", index)
	}
}

func TestFindString_StartIndex(t *testing.T) {
	re := regexp.MustCompile(`^test`)
	v := []string{"abc", "test1", "def", "test2"}
	index := findString(re, v, 2) // Start searching from index 2
	if index != 3 {
		t.Errorf("Expected index 3, got %d", index)
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

// Test case with float64 values
func TestGetRanges_Float64(t *testing.T) {
	f := func(x float64) bool { return x > 0 }
	v := []float64{-1.0, 2.5, 3.1, -3.1, 1.2}

	expected := []Range{{i: 1, j: 3}, {i: 4, j: 5}}
	result := getRanges(f, v)

	if !areEqualRanges(result, expected) {
		t.Errorf("TestGetRanges_Float64 failed: expected %v, got %v", expected, result)
	}
}

// Test case with integers
func TestGetRanges_Int(t *testing.T) {
	f := func(x int) bool { return x%2 == 0 } // Test for even numbers
	v := []int{1, 2, 4, 1, 6, 8, 3}

	expected := []Range{{i: 1, j: 3}, {i: 4, j: 6}}
	result := getRanges(f, v)

	if !areEqualRanges(result, expected) {
		t.Errorf("TestGetRanges_Int failed: expected %v, got %v", expected, result)
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

// Test case with strings
func TestGetRanges_String(t *testing.T) {
	f := func(s string) bool { return len(s) > 3 } // Test for strings longer than 3 characters
	v := []string{"go", "hello", "world", "hi", "golang"}

	expected := []Range{{i: 1, j: 3}, {i: 4, j: 5}}
	result := getRanges(f, v)

	if !areEqualRanges(result, expected) {
		t.Errorf("TestGetRanges_String failed: expected %v, got %v", expected, result)
	}
}

// Unit test for indentation function
func TestIndentation(t *testing.T) {
	tests := []struct {
		name     string
		s        string
		expected string
	}{
		{"Spaces", "    abc", "    "},
		{"Tabs", "\t\tabc", "\t\t"},
		{"Mixed spaces and tabs", " \t abc", " \t "},
		{"No indentation", "abc", ""},
		{"Only spaces", "    ", "    "},
		{"Only tabs", "\t\t", "\t\t"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := indentation(tt.s)
			if result != tt.expected {
				t.Errorf("Expected %q, got %q", tt.expected, result)
			}
		})
	}
}

// Unit test for indentations function
func TestIndentations(t *testing.T) {
	tests := []struct {
		name     string
		v        []string
		expected []string
	}{
		{"Multiple lines with different indentations", []string{"    line1", "\tline2", " line3"}, []string{"    ", "\t", " "}},
		{"Duplicate indentations", []string{"    line1", "    line2", "\tline3"}, []string{"    ", "\t"}},
		{"No indentation", []string{"line1", "line2"}, []string{""}},
		{"Empty slice", []string{}, []string{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := indentations(tt.v)
			if !eqStrings(result, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, result)
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

func TestJoinChunks(t *testing.T) {
	tests := []struct {
		name     string
		chunks   []Chunk
		expected []string
	}{
		{
			name: "Single chunk with multiple lines",
			chunks: []Chunk{
				{name: "Chunk1", lines: []string{"line1", "line2", "line3"}},
			},
			expected: []string{"line1", "line2", "line3"},
		},
		{
			name: "Multiple chunks with lines",
			chunks: []Chunk{
				{name: "Chunk1", lines: []string{"line1", "line2"}},
				{name: "Chunk2", lines: []string{"line3", "line4"}},
			},
			expected: []string{"line1", "line2", "line3", "line4"},
		},
		{
			name: "Empty chunk list",
			chunks: []Chunk{
				{name: "Chunk1", lines: nil},
			},
			expected: nil, // Expect nil here
		},
		{
			name: "Chunks with no lines",
			chunks: []Chunk{
				{name: "Chunk1", lines: []string{}},
				{name: "Chunk2", lines: []string{}},
			},
			expected: nil,
		},
		{
			name:     "No chunks",
			chunks:   []Chunk{},
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := joinChunks(tt.chunks)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Test function for parseCases
func TestParseCases(t *testing.T) {
	tests := []struct {
		name  string
		lines []string
		want  []Chunk
	}{
		{
			name: "Single case block",
			lines: []string{
				"switch x {",
				"case 1:",
				"    // Do something",
				"}",
			},
			want: []Chunk{
				createChunk("", "switch x {"),
				createChunk("case 1:", "case 1:", "    // Do something"),
				createChunk("", "}"),
			},
		},
		{
			name: "Multiple case blocks",
			lines: []string{
				"switch x {",
				"case 1:",
				"    // Do something",
				"case 2:",
				"    // Do something else",
				"}",
			},
			want: []Chunk{
				createChunk("", "switch x {"),
				createChunk("case 1:", "case 1:", "    // Do something"),
				createChunk("case 2:", "case 2:", "    // Do something else"),
				createChunk("", "}"),
			},
		},
		{
			name: "Default case",
			lines: []string{
				"switch x {",
				"default:",
				"    // Default case",
				"}",
			},
			want: []Chunk{
				createChunk("", "switch x {"),
				createChunk("default:", "default:", "    // Default case"),
				createChunk("", "}"),
			},
		},
		{
			name: "Multiple case labels in one block",
			lines: []string{
				"switch x {",
				"case 1:",
				"case 2:",
				"    // Do something for both cases",
				"}",
			},
			want: []Chunk{
				createChunk("", "switch x {"),
				createChunk("case 1:", "case 1:", "case 2:", "    // Do something for both cases"),
				createChunk("", "}"),
			},
		},
		{
			name: "No case blocks",
			lines: []string{
				"var x = 10;",
				"var y = 20;",
			},
			want: []Chunk{
				createChunk("", "var x = 10;", "var y = 20;"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseCases(tt.lines)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseCases() = %v, want %v", got, tt.want)
			}
		})
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

// Test case 1: Empty chunks
func TestSortChunks_Empty(t *testing.T) {
	chunks := []Chunk{}

	expected := []Chunk{}
	sortChunks(chunks)

	if !reflect.DeepEqual(chunks, expected) {
		t.Errorf("TestSortChunks_Empty failed: expected %v, got %v", expected, chunks)
	}
}

// Test case 4: Mixed special and non-special chunks
func TestSortChunks_MixedSpecial(t *testing.T) {
	chunks := []Chunk{
		{name: "", lines: []string{}},       // Non-special
		{name: "chunkC", lines: []string{}}, // Special
		{name: "chunkA", lines: []string{}}, // Special
		{name: "chunkB", lines: []string{}}, // Special
		{name: "", lines: []string{}},       // Non-special
	}

	expected := []Chunk{
		{name: "", lines: []string{}},       // Non-special
		{name: "chunkA", lines: []string{}}, // Sorted special
		{name: "chunkB", lines: []string{}}, // Sorted special
		{name: "chunkC", lines: []string{}}, // Sorted special
		{name: "", lines: []string{}},       // Non-special
	}
	sortChunks(chunks)

	if !reflect.DeepEqual(chunks, expected) {
		t.Errorf("TestSortChunks_MixedSpecial failed: expected %v, got %v", expected, chunks)
	}
}

// Test case 5: Multiple special ranges
func TestSortChunks_MultipleSpecialRanges(t *testing.T) {
	chunks := []Chunk{
		{name: "chunkC", lines: []string{}}, // Special
		{name: "chunkA", lines: []string{}}, // Special
		{name: "", lines: []string{}},       // Non-special
		{name: "chunkD", lines: []string{}}, // Special
		{name: "chunkB", lines: []string{}}, // Special
		{name: "", lines: []string{}},       // Non-special
	}

	expected := []Chunk{
		{name: "chunkA", lines: []string{}}, // Sorted special range 1
		{name: "chunkC", lines: []string{}}, // Sorted special range 1
		{name: "", lines: []string{}},       // Non-special
		{name: "chunkB", lines: []string{}}, // Sorted special range 2
		{name: "chunkD", lines: []string{}}, // Sorted special range 2
		{name: "", lines: []string{}},       // Non-special
	}
	sortChunks(chunks)

	if !reflect.DeepEqual(chunks, expected) {
		t.Errorf("TestSortChunks_MultipleSpecialRanges failed: expected %v, got %v", expected, chunks)
	}
}

// Test case 2: No special chunks (chunks with empty names)
func TestSortChunks_NoSpecial(t *testing.T) {
	chunks := []Chunk{
		{name: "", lines: []string{}},
		{name: "", lines: []string{}},
	}

	expected := []Chunk{
		{name: "", lines: []string{}},
		{name: "", lines: []string{}},
	}
	sortChunks(chunks)

	if !reflect.DeepEqual(chunks, expected) {
		t.Errorf("TestSortChunks_NoSpecial failed: expected %v, got %v", expected, chunks)
	}
}

// Test case 3: One special chunk range (with sorting needed)
func TestSortChunks_OneSpecialRange(t *testing.T) {
	chunks := []Chunk{
		{name: "chunkC", lines: []string{}},
		{name: "chunkA", lines: []string{}},
		{name: "chunkB", lines: []string{}},
	}

	expected := []Chunk{
		{name: "chunkA", lines: []string{}},
		{name: "chunkB", lines: []string{}},
		{name: "chunkC", lines: []string{}},
	}
	sortChunks(chunks)

	if !reflect.DeepEqual(chunks, expected) {
		t.Errorf("TestSortChunks_OneSpecialRange failed: expected %v, got %v", expected, chunks)
	}
}

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

// Helper function to compare slices of Range, accounting for nil vs empty slice differences
func areEqualRanges(a, b []Range) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

// Helper function to create a chunk
func createChunk(name string, lines ...string) Chunk {
	return Chunk{
		name:  name,
		lines: lines,
	}
}
