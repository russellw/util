import java.io.File

sealed class Token {
    data class LParen(val value: String = "(") : Token()
    data class RParen(val value: String = ")") : Token()
    data class Symbol(val value: String) : Token()
    data class Number(val value: Int) : Token()
    object EOF : Token()
}

sealed class Value {
    data class Integer(val value: Int) : Value()
    data class Symbol(val value: String) : Value()
    data class List(val values: kotlin.collections.List<Value>) : Value()
    data class Function(val eval: (Environment, kotlin.collections.List<Value>) -> Value)
}

class Lexer(private val input: String) {
    private var pos = 0
    private val tokens = mutableListOf<Token>()

    fun tokenize(): List<Token> {
        while (pos < input.length) {
            when (val c = input[pos]) {
                '(' -> {
                    tokens.add(Token.LParen())
                    pos++
                }
                ')' -> {
                    tokens.add(Token.RParen())
                    pos++
                }
                in " \t\n" -> pos++
                in '0'..'9' -> readNumber()
                else -> readSymbol()
            }
        }
        tokens.add(Token.EOF)
        return tokens
    }

    private fun readNumber() {
        val start = pos
        while (pos < input.length && input[pos] in '0'..'9') pos++
        tokens.add(Token.Number(input.substring(start, pos).toInt()))
    }

    private fun readSymbol() {
        val start = pos
        while (pos < input.length && input[pos] !in " \t\n()") pos++
        tokens.add(Token.Symbol(input.substring(start, pos)))
    }
}

class Parser(private val tokens: List<Token>) {
    private var pos = 0

    fun parse(): Value {
        return when (val token = tokens[pos++]) {
            is Token.LParen -> {
                val list = mutableListOf<Value>()
                while (tokens[pos] !is Token.RParen) {
                    if (tokens[pos] is Token.EOF) throw Exception("Unexpected EOF")
                    list.add(parse())
                }
                pos++ // consume RParen
                Value.List(list)
            }
            is Token.Number -> Value.Integer(token.value)
            is Token.Symbol -> Value.Symbol(token.value)
            is Token.RParen -> throw Exception("Unexpected )")
            is Token.EOF -> throw Exception("Unexpected EOF")
        }
    }
}

class Environment(private val parent: Environment? = null) {
    private val vars = mutableMapOf<String, Value>()

    init {
        vars["+"] = Value.Function { env, args ->
            Value.Integer(args.sumOf {
                when (val evaluated = env.eval(it)) {
                    is Value.Integer -> evaluated.value
                    else -> throw Exception("+ requires numbers")
                }
            })
        }

        vars["-"] = Value.Function { env, args ->
            if (args.isEmpty()) return@Function Value.Integer(0)
            val first = when (val evaluated = env.eval(args[0])) {
                is Value.Integer -> evaluated.value
                else -> throw Exception("- requires numbers")
            }
            Value.Integer(args.drop(1).fold(first) { acc, arg ->
                when (val evaluated = env.eval(arg)) {
                    is Value.Integer -> acc - evaluated.value
                    else -> throw Exception("- requires numbers")
                }
            })
        }

        vars["*"] = Value.Function { env, args ->
            Value.Integer(args.fold(1) { acc, arg ->
                when (val evaluated = env.eval(arg)) {
                    is Value.Integer -> acc * evaluated.value
                    else -> throw Exception("* requires numbers")
                }
            })
        }

        vars["<"] = Value.Function { env, args ->
            if (args.size != 2) throw Exception("< requires 2 arguments")
            val n1 = when (val e1 = env.eval(args[0])) {
                is Value.Integer -> e1.value
                else -> throw Exception("< requires numbers")
            }
            val n2 = when (val e2 = env.eval(args[1])) {
                is Value.Integer -> e2.value
                else -> throw Exception("< requires numbers")
            }
            Value.Integer(if (n1 < n2) 1 else 0)
        }

        vars["if"] = Value.Function { env, args ->
            if (args.size != 3) throw Exception("if requires 3 arguments")
            val cond = when (val evaluated = env.eval(args[0])) {
                is Value.Integer -> evaluated.value != 0
                else -> throw Exception("if condition must evaluate to a number")
            }
            env.eval(if (cond) args[1] else args[2])
        }

        vars["defun"] = Value.Function { env, args ->
            if (args.size < 3) throw Exception("defun requires at least 3 arguments")
            val name = when (val sym = args[0]) {
                is Value.Symbol -> sym.value
                else -> throw Exception("defun requires a symbol as first argument")
            }
            val params = when (val paramList = args[1]) {
                is Value.List -> paramList.values.map {
                    when (it) {
                        is Value.Symbol -> it.value
                        else -> throw Exception("defun parameters must be symbols")
                    }
                }
                else -> throw Exception("defun requires a parameter list")
            }
            val body = args[2]

            val fn = Value.Function { callEnv, callArgs ->
                if (callArgs.size != params.size) {
                    throw Exception("function $name expects ${params.size} arguments, got ${callArgs.size}")
                }
                val newEnv = Environment(callEnv)
                params.zip(callArgs).forEach { (param, arg) ->
                    newEnv.vars[param] = callEnv.eval(arg)
                }
                newEnv.eval(body)
            }
            env.vars[name] = fn
            Value.Symbol("Function $name defined")
        }
    }

    fun eval(expr: Value): Value = when (expr) {
        is Value.Integer -> expr
        is Value.Symbol -> get(expr.value) ?: throw Exception("undefined symbol: ${expr.value}")
        is Value.List -> {
            if (expr.values.isEmpty()) return expr
            when (val fn = eval(expr.values[0])) {
                is Value.Function -> fn.eval(this, expr.values.drop(1))
                else -> throw Exception("not a function")
            }
        }
        is Value.Function -> expr
    }

    private fun get(name: String): Value? =
        vars[name] ?: parent?.get(name)
}

fun main(args: Array<String>) {
    val interpreter = Environment()

    if (args.isNotEmpty()) {
        val content = File(args[0]).readText()
        evalString(interpreter, content)
    } else {
        while (true) {
            print("miniLisp> ")
            val line = readLine() ?: break
            if (line == "quit") break
            try {
                evalString(interpreter, line)
            } catch (e: Exception) {
                println("Error: ${e.message}")
            }
        }
    }
}

fun evalString(interpreter: Environment, input: String) {
    val lexer = Lexer(input)
    val tokens = lexer.tokenize()
    val parser = Parser(tokens)

    try {
        while (parser.pos < tokens.size - 1) {  // -1 to skip EOF
            val expr = parser.parse()
            val result = interpreter.eval(expr)
            if (result !is Value.Symbol || !result.value.startsWith("Function")) {
                println(when (result) {
                    is Value.Integer -> result.value
                    is Value.Symbol -> result.value
                    is Value.List -> result.values.joinToString(" ", "(", ")")
                    is Value.Function -> "<function>"
                })
            } else {
                println(result.value)
            }
        }
    } catch (e: Exception) {
        println("Error: ${e.message}")
    }
}
