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
