package main

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Value represents a TinyLisp value (integer, symbol, list, or function).
type Value interface{}

// Environment stores variable bindings.
type Environment map[string]Value

// Built-in functions.
var builtins = map[string]func(Value, *Environment) (Value, error){
	"add": func(args Value, env *Environment) (Value, error) {
		list, ok := args.([]Value)
		if !ok || len(list) != 2 {
			return nil, errors.New("add expects two arguments")
		}
		x, y := list[0], list[1]
		return x.(int) + y.(int), nil
	},
	"sub": func(args Value, env *Environment) (Value, error) {
		list, ok := args.([]Value)
		if !ok || len(list) != 2 {
			return nil, errors.New("sub expects two arguments")
		}
		x, y := list[0], list[1]
		return x.(int) - y.(int), nil
	},
	"mul": func(args Value, env *Environment) (Value, error) {
		list, ok := args.([]Value)
		if !ok || len(list) != 2 {
			return nil, errors.New("mul expects two arguments")
		}
		x, y := list[0], list[1]
		return x.(int) * y.(int), nil
	},
	"div": func(args Value, env *Environment) (Value, error) {
		list, ok := args.([]Value)
		if !ok || len(list) != 2 {
			return nil, errors.New("div expects two arguments")
		}
		x, y := list[0], list[1]
		return x.(int) / y.(int), nil
	},
	"eq": func(args Value, env *Environment) (Value, error) {
		list, ok := args.([]Value)
		if !ok || len(list) != 2 {
			return nil, errors.New("eq expects two arguments")
		}
		x, y := list[0], list[1]
		if x == y {
			return 1, nil
		}
		return 0, nil
	},
}

// Eval evaluates a TinyLisp expression.
func Eval(expr Value, env *Environment) (Value, error) {
	switch v := expr.(type) {
	case int: // Integer literal
		return v, nil
	case string: // Symbol
		if val, ok := (*env)[v]; ok {
			return val, nil
		}
		return nil, fmt.Errorf("undefined symbol: %s", v)
	case []Value: // List (function call or special form)
		if len(v) == 0 {
			return nil, errors.New("empty list")
		}
		first := v[0]
		switch first {
		case "define": // (define x expr)
			if len(v) != 3 {
				return nil, errors.New("define expects two arguments")
			}
			sym, ok := v[1].(string)
			if !ok {
				return nil, errors.New("define expects a symbol")
			}
			val, err := Eval(v[2], env)
			if err != nil {
				return nil, err
			}
			(*env)[sym] = val
			return val, nil
		case "if": // (if cond then else)
			if len(v) != 4 {
				return nil, errors.New("if expects three arguments")
			}
			cond, err := Eval(v[1], env)
			if err != nil {
				return nil, err
			}
			if cond.(int) != 0 {
				return Eval(v[2], env)
			}
			return Eval(v[3], env)
		default: // Function call
			fn, err := Eval(first, env)
			if err != nil {
				return nil, err
			}
			args := v[1:]
			evaluatedArgs := make([]Value, len(args))
			for i, arg := range args {
				evaluatedArg, err := Eval(arg, env)
				if err != nil {
					return nil, err
				}
				evaluatedArgs[i] = evaluatedArg
			}
			if f, ok := fn.(func(Value, *Environment) (Value, error)); ok {
				return f(evaluatedArgs, env)
			}
			return nil, fmt.Errorf("not a function: %v", first)
		}
	default:
		return nil, fmt.Errorf("unknown expression type: %v", expr)
	}
}

// Parse parses a TinyLisp expression from a string.
func Parse(input string) (Value, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return nil, errors.New("empty input")
	}
	if input[0] == '(' {
		// Parse a list
		input = input[1 : len(input)-1]
		tokens := strings.Fields(input)
		list := make([]Value, len(tokens))
		for i, token := range tokens {
			val, err := Parse(token)
			if err != nil {
				return nil, err
			}
			list[i] = val
		}
		return list, nil
	}
	// Parse an integer or symbol
	if num, err := strconv.Atoi(input); err == nil {
		return num, nil
	}
	return input, nil // Symbol
}

func main() {
	env := &Environment{}
	for k, v := range builtins {
		(*env)[k] = v
	}

	// Example usage
	expr := "(add 1 (mul 2 3))"
	parsed, err := Parse(expr)
	if err != nil {
		fmt.Println("Parse error:", err)
		return
	}
	result, err := Eval(parsed, env)
	if err != nil {
		fmt.Println("Eval error:", err)
		return
	}
	fmt.Println("Result:", result) // Output: Result: 7
}
