package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
)

type TokenType int

const (
	LPAREN TokenType = iota
	RPAREN
	SYMBOL
	NUMBER
	EOF
)

type Token struct {
	Type  TokenType
	Value string
}

type Lexer struct {
	input  string
	pos    int
	tokens []Token
}

func NewLexer(input string) *Lexer {
	return &Lexer{input: input, pos: 0}
}

func (l *Lexer) tokenize() []Token {
	for l.pos < len(l.input) {
		switch c := l.input[l.pos]; {
		case c == '(':
			l.tokens = append(l.tokens, Token{LPAREN, "("})
			l.pos++
		case c == ')':
			l.tokens = append(l.tokens, Token{RPAREN, ")"})
			l.pos++
		case c == ' ' || c == '\t' || c == '\n':
			l.pos++
		default:
			if isDigit(c) {
				l.readNumber()
			} else {
				l.readSymbol()
			}
		}
	}
	l.tokens = append(l.tokens, Token{EOF, ""})
	return l.tokens
}

func (l *Lexer) readNumber() {
	start := l.pos
	for l.pos < len(l.input) && isDigit(l.input[l.pos]) {
		l.pos++
	}
	l.tokens = append(l.tokens, Token{NUMBER, l.input[start:l.pos]})
}

func (l *Lexer) readSymbol() {
	start := l.pos
	for l.pos < len(l.input) && !isDelimiter(l.input[l.pos]) {
		l.pos++
	}
	l.tokens = append(l.tokens, Token{SYMBOL, l.input[start:l.pos]})
}

func isDigit(c byte) bool {
	return c >= '0' && c <= '9'
}

func isDelimiter(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '(' || c == ')'
}

type Value interface{}
type Symbol string
type List []Value
type Function func(*Environment, []Value) (Value, error)

type Parser struct {
	tokens []Token
	pos    int
}

func NewParser(tokens []Token) *Parser {
	return &Parser{tokens: tokens}
}

func (p *Parser) parse() (Value, error) {
	token := p.tokens[p.pos]
	switch token.Type {
	case LPAREN:
		p.pos++
		list := List{}
		for p.tokens[p.pos].Type != RPAREN {
			if p.tokens[p.pos].Type == EOF {
				return nil, fmt.Errorf("unexpected EOF")
			}
			val, err := p.parse()
			if err != nil {
				return nil, err
			}
			list = append(list, val)
		}
		p.pos++ // consume RPAREN
		return list, nil
	case NUMBER:
		p.pos++
		n, _ := strconv.Atoi(token.Value)
		return n, nil
	case SYMBOL:
		p.pos++
		return Symbol(token.Value), nil
	default:
		return nil, fmt.Errorf("unexpected token: %v", token)
	}
}

type Environment struct {
	vars   map[Symbol]Value
	parent *Environment
}

func NewEnvironment() *Environment {
	env := &Environment{
		vars: make(map[Symbol]Value),
	}

	env.vars["+"] = Function(func(env *Environment, args []Value) (Value, error) {
		result := 0
		for _, arg := range args {
			evaluated, err := env.Eval(arg)
			if err != nil {
				return nil, err
			}
			n, ok := evaluated.(int)
			if !ok {
				return nil, fmt.Errorf("'+' requires numbers")
			}
			result += n
		}
		return result, nil
	})

	env.vars["-"] = Function(func(env *Environment, args []Value) (Value, error) {
		if len(args) == 0 {
			return 0, nil
		}
		first, err := env.Eval(args[0])
		if err != nil {
			return nil, err
		}
		firstNum, ok := first.(int)
		if !ok {
			return nil, fmt.Errorf("'-' requires numbers")
		}
		result := firstNum
		for _, arg := range args[1:] {
			evaluated, err := env.Eval(arg)
			if err != nil {
				return nil, err
			}
			n, ok := evaluated.(int)
			if !ok {
				return nil, fmt.Errorf("'-' requires numbers")
			}
			result -= n
		}
		return result, nil
	})

	env.vars["*"] = Function(func(env *Environment, args []Value) (Value, error) {
		result := 1
		for _, arg := range args {
			evaluated, err := env.Eval(arg)
			if err != nil {
				return nil, err
			}
			n, ok := evaluated.(int)
			if !ok {
				return nil, fmt.Errorf("'*' requires numbers")
			}
			result *= n
		}
		return result, nil
	})

	env.vars["if"] = Function(func(env *Environment, args []Value) (Value, error) {
		if len(args) != 3 {
			return nil, fmt.Errorf("if requires 3 arguments")
		}
		cond, err := env.Eval(args[0])
		if err != nil {
			return nil, err
		}
		condVal, ok := cond.(int)
		if !ok {
			return nil, fmt.Errorf("if condition must evaluate to a number")
		}
		if condVal != 0 {
			return env.Eval(args[1])
		}
		return env.Eval(args[2])
	})

	env.vars["<"] = Function(func(env *Environment, args []Value) (Value, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("< requires 2 arguments")
		}
		a1, err := env.Eval(args[0])
		if err != nil {
			return nil, err
		}
		a2, err := env.Eval(args[1])
		if err != nil {
			return nil, err
		}
		n1, ok1 := a1.(int)
		n2, ok2 := a2.(int)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("< requires numbers")
		}
		if n1 < n2 {
			return 1, nil
		}
		return 0, nil
	})

	env.vars["defun"] = Function(func(env *Environment, args []Value) (Value, error) {
		if len(args) < 3 {
			return nil, fmt.Errorf("defun requires at least 3 arguments")
		}
		name, ok := args[0].(Symbol)
		if !ok {
			return nil, fmt.Errorf("defun requires a symbol as first argument")
		}
		params, ok := args[1].(List)
		if !ok {
			return nil, fmt.Errorf("defun requires a parameter list")
		}
		paramNames := make([]Symbol, len(params))
		for i, p := range params {
			sym, ok := p.(Symbol)
			if !ok {
				return nil, fmt.Errorf("defun parameters must be symbols")
			}
			paramNames[i] = sym
		}

		body := args[2]
		fn := Function(func(env *Environment, args []Value) (Value, error) {
			if len(args) != len(paramNames) {
				return nil, fmt.Errorf("function %s expects %d arguments, got %d", name, len(paramNames), len(args))
			}
			newEnv := NewEnvironment()
			newEnv.parent = env
			for i, param := range paramNames {
				evaluated, err := env.Eval(args[i])
				if err != nil {
					return nil, err
				}
				newEnv.vars[param] = evaluated
			}
			return newEnv.Eval(body)
		})

		env.vars[name] = fn
		return fmt.Sprintf("Function %s defined", name), nil
	})

	return env
}

func (env *Environment) Get(sym Symbol) (Value, bool) {
	if val, ok := env.vars[sym]; ok {
		return val, true
	}
	if env.parent != nil {
		return env.parent.Get(sym)
	}
	return nil, false
}

func (env *Environment) Set(sym Symbol, val Value) {
	env.vars[sym] = val
}

func (env *Environment) Eval(expr Value) (Value, error) {
	switch v := expr.(type) {
	case int:
		return v, nil
	case Symbol:
		if val, ok := env.Get(v); ok {
			return val, nil
		}
		return nil, fmt.Errorf("undefined symbol: %v", v)
	case List:
		if len(v) == 0 {
			return nil, nil
		}
		fn, err := env.Eval(v[0])
		if err != nil {
			return nil, err
		}
		f, ok := fn.(Function)
		if !ok {
			return nil, fmt.Errorf("not a function")
		}
		return f(env, v[1:])
	default:
		return nil, fmt.Errorf("unknown expression type: %T", expr)
	}
}

func evalString(interpreter *Environment, input string) error {
	lexer := NewLexer(input)
	tokens := lexer.tokenize()
	parser := NewParser(tokens)

	for parser.pos < len(parser.tokens)-1 {  // -1 to skip final EOF
		expr, err := parser.parse()
		if err != nil {
			return fmt.Errorf("Parse error: %v", err)
		}

		result, err := interpreter.Eval(expr)
		if err != nil {
			return fmt.Errorf("Eval error: %v", err)
		}
		if result != nil {
			fmt.Println(result)
		}
	}
	return nil
}

func main() {
	interpreter := NewEnvironment()

	if len(os.Args) > 1 {
		content, err := os.ReadFile(os.Args[1])
		if err != nil {
			fmt.Printf("Error reading file: %v\n", err)
			os.Exit(1)
		}
		if err := evalString(interpreter, string(content)); err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		return
	}

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Print("miniLisp> ")

	for scanner.Scan() {
		input := scanner.Text()
		if input == "quit" {
			break
		}

		if err := evalString(interpreter, input); err != nil {
			fmt.Println(err)
		}
		fmt.Print("miniLisp> ")
	}
}