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
type Function func([]Value) (Value, error)

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

type Environment map[Symbol]Function

func NewEnvironment() Environment {
	env := Environment{}
	
	env["+"] = func(args []Value) (Value, error) {
		result := 0
		for _, arg := range args {
			n, ok := arg.(int)
			if !ok {
				return nil, fmt.Errorf("'+' requires numbers")
			}
			result += n
		}
		return result, nil
	}

	env["*"] = func(args []Value) (Value, error) {
		result := 1
		for _, arg := range args {
			n, ok := arg.(int)
			if !ok {
				return nil, fmt.Errorf("'*' requires numbers")
			}
			result *= n
		}
		return result, nil
	}

	env["-"] = func(args []Value) (Value, error) {
		if len(args) == 0 {
			return 0, nil
		}
		first, ok := args[0].(int)
		if !ok {
			return nil, fmt.Errorf("'-' requires numbers")
		}
		result := first
		for _, arg := range args[1:] {
			n, ok := arg.(int)
			if !ok {
				return nil, fmt.Errorf("'-' requires numbers")
			}
			result -= n
		}
		return result, nil
	}

	env["list"] = func(args []Value) (Value, error) {
		return List(args), nil
	}

	env["car"] = func(args []Value) (Value, error) {
		if len(args) != 1 {
			return nil, fmt.Errorf("car takes exactly one argument")
		}
		list, ok := args[0].(List)
		if !ok {
			return nil, fmt.Errorf("car requires a list")
		}
		if len(list) == 0 {
			return nil, fmt.Errorf("car: empty list")
		}
		return list[0], nil
	}

	env["cdr"] = func(args []Value) (Value, error) {
		if len(args) != 1 {
			return nil, fmt.Errorf("cdr takes exactly one argument")
		}
		list, ok := args[0].(List)
		if !ok {
			return nil, fmt.Errorf("cdr requires a list")
		}
		if len(list) <= 1 {
			return List{}, nil
		}
		return list[1:], nil
	}

	env["cons"] = func(args []Value) (Value, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("cons takes exactly two arguments")
		}
		list, ok := args[1].(List)
		if !ok {
			return nil, fmt.Errorf("cons requires a list as second argument")
		}
		return append(List{args[0]}, list...), nil
	}

	return env
}

type Interpreter struct {
	env Environment
}

func NewInterpreter() *Interpreter {
	return &Interpreter{env: NewEnvironment()}
}

func (i *Interpreter) Eval(expr Value) (Value, error) {
	switch v := expr.(type) {
	case int:
		return v, nil
	case Symbol:
		if fn, ok := i.env[v]; ok {
			return fn, nil
		}
		return nil, fmt.Errorf("undefined symbol: %v", v)
	case List:
		if len(v) == 0 {
			return nil, nil
		}
		fn, err := i.Eval(v[0])
		if err != nil {
			return nil, err
		}
		f, ok := fn.(Function)
		if !ok {
			return nil, fmt.Errorf("not a function")
		}
		args := make([]Value, 0, len(v)-1)
		for _, arg := range v[1:] {
			evaledArg, err := i.Eval(arg)
			if err != nil {
				return nil, err
			}
			args = append(args, evaledArg)
		}
		return f(args)
	default:
		return nil, fmt.Errorf("unknown expression type: %T", expr)
	}
}

func evalString(interpreter *Interpreter, input string) error {
	lexer := NewLexer(input)
	tokens := lexer.tokenize()
	parser := NewParser(tokens)
	
	expr, err := parser.parse()
	if err != nil {
		return fmt.Errorf("Parse error: %v", err)
	}
	
	result, err := interpreter.Eval(expr)
	if err != nil {
		return fmt.Errorf("Eval error: %v", err)
	}
	fmt.Println(result)
	return nil
}

func main() {
	interpreter := NewInterpreter()

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