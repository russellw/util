package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Expr is the type for all Lisp expressions.
type Expr interface{}

// Symbol represents a Lisp symbol.
type Symbol string

// Number represents a numeric atom.
type Number float64

// List represents a Lisp list (S-expression).
type List []Expr

// Function is the type for callable functions in our Lisp.
type Function func(args []Expr) (Expr, error)

// Env represents the environment mapping symbols to their values.
type Env map[string]Expr

// tokenize converts the input string into a slice of tokens.
func tokenize(input string) []string {
	// Add spaces around parentheses so that they become separate tokens.
	input = strings.ReplaceAll(input, "(", " ( ")
	input = strings.ReplaceAll(input, ")", " ) ")
	return strings.Fields(input)
}

// parse consumes tokens and returns the corresponding Expr.
func parse(tokens []string) (Expr, []string, error) {
	if len(tokens) == 0 {
		return nil, tokens, fmt.Errorf("unexpected EOF while reading")
	}
	token := tokens[0]
	tokens = tokens[1:]

	if token == "(" {
		var lst List
		// Continue parsing until a ")" is encountered.
		for len(tokens) > 0 && tokens[0] != ")" {
			var expr Expr
			var err error
			expr, tokens, err = parse(tokens)
			if err != nil {
				return nil, tokens, err
			}
			lst = append(lst, expr)
		}
		if len(tokens) == 0 {
			return nil, tokens, fmt.Errorf("expected ')'")
		}
		// Discard the ")"
		tokens = tokens[1:]
		return lst, tokens, nil
	} else if token == ")" {
		return nil, tokens, fmt.Errorf("unexpected ')'")
	} else {
		// An atom: try to convert it to a number; if not, treat it as a symbol.
		return atom(token), tokens, nil
	}
}

// atom converts a token into a Number (if possible) or a Symbol.
func atom(token string) Expr {
	if num, err := strconv.ParseFloat(token, 64); err == nil {
		return Number(num)
	}
	return Symbol(token)
}

// standardEnv returns an environment with some built-in functions.
func standardEnv() Env {
	env := make(Env)

	// Addition: (+ a b c ...)
	env["+"] = Function(func(args []Expr) (Expr, error) {
		sum := Number(0)
		for _, arg := range args {
			n, ok := arg.(Number)
			if !ok {
				return nil, fmt.Errorf("expected number, got %v", arg)
			}
			sum += n
		}
		return sum, nil
	})

	// Subtraction: (- a b c ...). With one argument, returns its negation.
	env["-"] = Function(func(args []Expr) (Expr, error) {
		if len(args) == 0 {
			return nil, fmt.Errorf("'-' expects at least one argument")
		}
		first, ok := args[0].(Number)
		if !ok {
			return nil, fmt.Errorf("expected number, got %v", args[0])
		}
		if len(args) == 1 {
			return -first, nil
		}
		result := first
		for _, arg := range args[1:] {
			n, ok := arg.(Number)
			if !ok {
				return nil, fmt.Errorf("expected number, got %v", arg)
			}
			result -= n
		}
		return result, nil
	})

	// Multiplication: (* a b c ...)
	env["*"] = Function(func(args []Expr) (Expr, error) {
		result := Number(1)
		for _, arg := range args {
			n, ok := arg.(Number)
			if !ok {
				return nil, fmt.Errorf("expected number, got %v", arg)
			}
			result *= n
		}
		return result, nil
	})

	// Division: (/ a b c ...). With one argument, returns 1 divided by it.
	env["/"] = Function(func(args []Expr) (Expr, error) {
		if len(args) == 0 {
			return nil, fmt.Errorf("'/' expects at least one argument")
		}
		first, ok := args[0].(Number)
		if !ok {
			return nil, fmt.Errorf("expected number, got %v", args[0])
		}
		if len(args) == 1 {
			if first == 0 {
				return nil, fmt.Errorf("division by zero")
			}
			return Number(1) / first, nil
		}
		result := first
		for _, arg := range args[1:] {
			n, ok := arg.(Number)
			if !ok {
				return nil, fmt.Errorf("expected number, got %v", arg)
			}
			if n == 0 {
				return nil, fmt.Errorf("division by zero")
			}
			result /= n
		}
		return result, nil
	})

	// You can add more built-in functions here as needed.

	return env
}

// eval recursively evaluates an expression in the given environment.
func eval(x Expr, env Env) (Expr, error) {
	switch exp := x.(type) {
	case Symbol:
		// Look up symbols in the environment.
		val, ok := env[string(exp)]
		if !ok {
			return nil, fmt.Errorf("undefined symbol: %s", exp)
		}
		return val, nil
	case Number:
		// Numbers evaluate to themselves.
		return exp, nil
	case List:
		if len(exp) == 0 {
			return nil, fmt.Errorf("cannot evaluate an empty list")
		}
		// Check for special forms first.
		switch first := exp[0].(type) {
		case Symbol:
			switch first {
			case "if":
				// (if test consequent alternate)
				if len(exp) != 4 {
					return nil, fmt.Errorf("if expects 3 arguments")
				}
				test, err := eval(exp[1], env)
				if err != nil {
					return nil, err
				}
				// Here we consider a number unequal to zero as true.
				isTrue := false
				if n, ok := test.(Number); ok {
					isTrue = n != 0
				} else {
					isTrue = test != nil
				}
				if isTrue {
					return eval(exp[2], env)
				}
				return eval(exp[3], env)

			case "define":
				// (define symbol expr)
				if len(exp) != 3 {
					return nil, fmt.Errorf("define expects 2 arguments")
				}
				sym, ok := exp[1].(Symbol)
				if !ok {
					return nil, fmt.Errorf("define: first argument must be a symbol")
				}
				val, err := eval(exp[2], env)
				if err != nil {
					return nil, err
				}
				env[string(sym)] = val
				return val, nil

			case "lambda":
				// (lambda (params...) body)
				if len(exp) != 3 {
					return nil, fmt.Errorf("lambda expects 2 arguments")
				}
				paramsList, ok := exp[1].(List)
				if !ok {
					return nil, fmt.Errorf("lambda: parameters must be a list")
				}
				var params []string
				for _, param := range paramsList {
					psym, ok := param.(Symbol)
					if !ok {
						return nil, fmt.Errorf("lambda: parameter must be a symbol")
					}
					params = append(params, string(psym))
				}
				body := exp[2]
				// Return a closure capturing the current environment.
				closure := Function(func(args []Expr) (Expr, error) {
					if len(args) != len(params) {
						return nil, fmt.Errorf("expected %d arguments, got %d", len(params), len(args))
					}
					// Extend the environment for the function call.
					newEnv := make(Env)
					// Copy the outer environment.
					for k, v := range env {
						newEnv[k] = v
					}
					// Bind parameters to arguments.
					for i, param := range params {
						newEnv[param] = args[i]
					}
					return eval(body, newEnv)
				})
				return closure, nil
			}
		}
		// If not a special form, evaluate all elements in the list.
		var evaluated []Expr
		for _, expr := range exp {
			ev, err := eval(expr, env)
			if err != nil {
				return nil, err
			}
			evaluated = append(evaluated, ev)
		}
		// The first element should now be a function to call.
		fn, ok := evaluated[0].(Function)
		if !ok {
			return nil, fmt.Errorf("first element is not a function: %v", evaluated[0])
		}
		return fn(evaluated[1:])

	default:
		return nil, fmt.Errorf("unknown expression type: %v", exp)
	}
}

// repl implements a simple read–eval–print loop.
func repl() {
	env := standardEnv()
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("lisp> ")
		if !scanner.Scan() {
			break
		}
		line := scanner.Text()
		tokens := tokenize(line)
		if len(tokens) == 0 {
			continue
		}
		expr, remaining, err := parse(tokens)
		if err != nil {
			fmt.Println("Parse error:", err)
			continue
		}
		if len(remaining) > 0 {
			fmt.Println("Warning: unused tokens:", remaining)
		}
		result, err := eval(expr, env)
		if err != nil {
			fmt.Println("Evaluation error:", err)
			continue
		}
		fmt.Println("=>", result)
	}
}

func prettyPrint(expr Expr) string {
    switch v := expr.(type) {
    case Number:
        return fmt.Sprintf("%g", float64(v))
    case Symbol:
        return string(v)
    case List:
        var parts []string
        for _, e := range v {
            parts = append(parts, prettyPrint(e))
        }
        return "(" + strings.Join(parts, " ") + ")"
    case Function:
        // Instead of printing a pointer, show a friendly message.
        return "<function>"
    default:
        return fmt.Sprintf("%v", expr)
    }
}

func main() {
    env := standardEnv()
    if len(os.Args) > 1 {
        // Read the file.
        data, err := os.ReadFile(os.Args[1])
        if err != nil {
            fmt.Println("Error reading file:", err)
            return
        }
        tokens := tokenize(string(data))
        var result Expr
        // Parse and evaluate all top-level expressions.
        for len(tokens) > 0 {
            var expr Expr
            expr, tokens, err = parse(tokens)
            if err != nil {
                fmt.Println("Parse error:", err)
                return
            }
            result, err = eval(expr, env)
            if err != nil {
                fmt.Println("Evaluation error:", err)
                return
            }
        }
        fmt.Println(prettyPrint(result))
    } else {
        repl()
    }
}
