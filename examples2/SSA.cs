using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class Block : IList<Term>
{
    public string name;
    public List<Term> code = new List<Term>();

    public Block(string name)
    {
        this.name = name;
    }

    public Term this[int index] { get => ((IList<Term>)code)[index]; set => ((IList<Term>)code)[index] = value; }

    public int Count => ((IList<Term>)code).Count;

    public bool IsReadOnly => ((IList<Term>)code).IsReadOnly;

    public void Add(Term item)
    {
        ((IList<Term>)code).Add(item);
    }

    public void Clear()
    {
        ((IList<Term>)code).Clear();
    }

    public bool Contains(Term item)
    {
        return ((IList<Term>)code).Contains(item);
    }

    public void CopyTo(Term[] array, int arrayIndex)
    {
        ((IList<Term>)code).CopyTo(array, arrayIndex);
    }

    public IEnumerator<Term> GetEnumerator()
    {
        return ((IList<Term>)code).GetEnumerator();
    }

    public int IndexOf(Term item)
    {
        return ((IList<Term>)code).IndexOf(item);
    }

    public void Insert(int index, Term item)
    {
        ((IList<Term>)code).Insert(index, item);
    }

    public bool Remove(Term item)
    {
        return ((IList<Term>)code).Remove(item);
    }

    public void RemoveAt(int index)
    {
        ((IList<Term>)code).RemoveAt(index);
    }

    public override string ToString()
    {
        return name;
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return ((IList<Term>)code).GetEnumerator();
    }
}

class Program
{
    public readonly List<Term[]> asts = new List<Term[]>();
    public readonly List<Function> funcs = new List<Function>();

    public void ssa()
    {
        var main = new Function(null, "main", new Var[0], new List<Term>());
        foreach (var ast in asts)
        {
            main.body.AddRange(ast);
        }
        funcs.Add(main);
        convert(main);
    }

    static void convert(Function f)
    {
        f.blocks.Add(new Block("entry"));

        Term postOp(Env env, Term a, Func<Pos, Term, Term, Term> op)
        {
            if (!(a[0] is Id id))
                throw new Error(a[0].pos, "expected lvalue");
            var x = get(env, id.name);
            if (x == null)
                throw new Error(a.pos, "not found");
            var load = new Load(a.pos, x);
            f.blocks.Last().Add(load);
            var b = op(a.pos, load, new IntLiteral(a.pos, 1));
            f.blocks.Last().Add(b);
            f.blocks.Last().Add(new Store(a.pos, x, b));
            return load;
        }

        Term opAssign(Env env, Term a, Func<Pos, Term, Term, Term> op)
        {
            if (!(a[0] is Id id))
                throw new Error(a[0].pos, "expected lvalue");
            var x = get(env, id.name);
            if (x == null)
                throw new Error(a.pos, "not found");
            var load = new Load(a.pos, x);
            f.blocks.Last().Add(load);
            var b = op(a.pos, load, term(env, a[1]));
            f.blocks.Last().Add(b);
            f.blocks.Last().Add(new Store(a.pos, x, b));
            return b;
        }

        Term term(Env env, Term a0)
        {
            switch (a0)
            {
                case Or a:
                    return term(env, new Cond(a.pos, a[0], new True(a.pos), a[1]));
                case And a:
                    return term(env, new Cond(a.pos, a[0], a[1], new False(a.pos)));
                case Not a:
                    return term(env, new Cond(a.pos, a[0], new False(a.pos), new True(a.pos)));
                case PreInc a:
                    return term(env, new AddAssign(a.pos, a[0], new IntLiteral(a.pos, 1)));
                case PreDec a:
                    return term(env, new SubAssign(a.pos, a[0], new IntLiteral(a.pos, 1)));
                case PostInc a:
                    return postOp(env, a, (pos, x, y) => new Add(pos, x, y));
                case PostDec a:
                    return postOp(env, a, (pos, x, y) => new Sub(pos, x, y));
                case Assign a:
                    {
                        if (!(a[0] is Id id))
                            throw new Error(a[0].pos, "expected lvalue");
                        var x = get(env, id.name);
                        if (x == null)
                            throw new Error(a.pos, "not found");
                        var b = term(env, a[1]);
                        f.blocks.Last().Add(new Store(a.pos, x, b));
                        return b;
                    }
                case AddAssign a:
                    return opAssign(env, a, (pos, x, y) => new Add(pos, x, y));
                case SubAssign a:
                    return opAssign(env, a, (pos, x, y) => new Sub(pos, x, y));
                case MulAssign a:
                    return opAssign(env, a, (pos, x, y) => new Mul(pos, x, y));
                case DivAssign a:
                    return opAssign(env, a, (pos, x, y) => new Div(pos, x, y));
                case RemAssign a:
                    return opAssign(env, a, (pos, x, y) => new Rem(pos, x, y));
                case ShlAssign a:
                    return opAssign(env, a, (pos, x, y) => new Shl(pos, x, y));
                case ShrAssign a:
                    return opAssign(env, a, (pos, x, y) => new Shr(pos, x, y));
                case BitAndAssign a:
                    return opAssign(env, a, (pos, x, y) => new BitAnd(pos, x, y));
                case BitOrAssign a:
                    return opAssign(env, a, (pos, x, y) => new BitOr(pos, x, y));
                case BitXorAssign a:
                    return opAssign(env, a, (pos, x, y) => new BitXor(pos, x, y));
                case Id a:
                    {
                        var x = get(env, a.name);
                        if (x == null)
                            throw new Error(a.pos, "not found");
                        var load = new Load(a.pos, x);
                        f.blocks.Last().Add(load);
                        return load;
                    }
                case Var a:
                    put(env, a.name, a);
                    f.vars.Add(a);
                    if (a.Count > 0)
                    {
                        var b = term(env, a[0]);
                        f.blocks.Last().Add(new Store(a.pos, a, b));
                        return b;
                    }
                    return null;
                case Break a:
                    {
                        var target = breakTarget(env);
                        if (target == null)
                            throw new Error(a.pos, "break without loop");
                        f.blocks.Last().Add(new Goto(a.pos, target));
                        f.blocks.Add(new Block("after_break"));
                        return null;
                    }
                case Continue a:
                    {
                        var target = continueTarget(env);
                        if (target == null)
                            throw new Error(a.pos, "continue without loop");
                        f.blocks.Last().Add(new Goto(a.pos, target));
                        f.blocks.Add(new Block("after_continue"));
                        return null;
                    }
                case CompoundStmt a:
                    env = new Env(env, null, null);
                    foreach (var b in a)
                        term(env, b);
                    return null;
                case DoWhile a:
                    {
                        var body = new Block("dowhile_body");
                        var test = new Block("dowhile_test");
                        var after = new Block("dowhile_after");
                        env = new Env(env, test, after);

                        // before
                        f.blocks.Last().Add(new Goto(a.pos, body));

                        // body
                        f.blocks.Add(body);
                        term(env, a[0]);
                        f.blocks.Last().Add(new Goto(a.pos, test));

                        // test
                        f.blocks.Add(test);
                        var t = term(env, a[1]);
                        f.blocks.Last().Add(new Br(a.pos, t, body, after));

                        // after
                        f.blocks.Add(after);
                        return null;
                    }
                case While a:
                    {
                        var test = new Block("while_test");
                        var body = new Block("while_body");
                        var after = new Block("while_after");
                        env = new Env(env, test, after);

                        // before
                        f.blocks.Last().Add(new Goto(a.pos, test));

                        // test
                        f.blocks.Add(test);
                        var t = term(env, a[0]);
                        f.blocks.Last().Add(new Br(a.pos, t, body, after));

                        // body
                        f.blocks.Add(body);
                        term(env, a[1]);
                        f.blocks.Last().Add(new Goto(a.pos, test));

                        // after
                        f.blocks.Add(after);
                        return null;
                    }
                case For a:
                    {
                        var test = new Block("for_test");
                        var body = new Block("for_body");
                        var update = new Block("for_update");
                        var after = new Block("for_after");
                        env = new Env(env, update, after);

                        // init
                        term(env, a[0]);
                        f.blocks.Last().Add(new Goto(a.pos, test));

                        // test
                        f.blocks.Add(test);
                        var t = term(env, a[1]);
                        f.blocks.Last().Add(new Br(a.pos, t, body, after));

                        // body
                        f.blocks.Add(body);
                        term(env, a[3]);
                        f.blocks.Last().Add(new Goto(a.pos, update));

                        // update
                        f.blocks.Add(update);
                        term(env, a[2]);
                        f.blocks.Last().Add(new Goto(a.pos, test));

                        // after
                        f.blocks.Add(after);
                        return null;
                    }
                case If a:
                    {
                        var ifTrue = new Block("if_true");
                        var ifFalse = new Block("if_false");
                        var after = new Block("if_after");

                        // test
                        var t = term(env, a[0]);
                        f.blocks.Last().Add(new Br(a.pos, t, ifTrue, ifFalse));

                        // true
                        f.blocks.Add(ifTrue);
                        term(env, a[1]);
                        f.blocks.Last().Add(new Goto(a.pos, after));

                        // false
                        f.blocks.Add(ifFalse);
                        if (a.Count == 3)
                            term(env, a[2]);
                        f.blocks.Last().Add(new Goto(a.pos, after));

                        // after
                        f.blocks.Add(after);
                        return null;
                    }
                case Cond a:
                    {
                        var x = new Var(a.pos, "tmp", new TBool(), null);
                        f.vars.Add(x);
                        var ifTrue = new Block("cond_true");
                        var ifFalse = new Block("cond_false");
                        var after = new Block("cond_after");

                        // test
                        var t = term(env, a[0]);
                        f.blocks.Last().Add(new Br(a.pos, t, ifTrue, ifFalse));

                        // true
                        f.blocks.Add(ifTrue);
                        f.blocks.Last().Add(new Store(a.pos, x, term(env, a[1])));
                        f.blocks.Last().Add(new Goto(a.pos, after));

                        // false
                        f.blocks.Add(ifFalse);
                        f.blocks.Last().Add(new Store(a.pos, x, term(env, a[2])));
                        f.blocks.Last().Add(new Goto(a.pos, after));

                        // after
                        f.blocks.Add(after);
                        var load = new Load(a.pos, x);
                        f.blocks.Last().Add(load);
                        return load;
                    }
                case Goto a:
                    f.blocks.Last().Add(a);
                    f.blocks.Add(new Block("after_goto_" + a.label));
                    return null;
                case Label a:
                    var block = new Block(a.name);
                    f.blocks.Last().Add(new Goto(a.pos, block));
                    f.blocks.Add(block);
                    f.labels.Add(a.name, block);
                    return null;
                case Return a:
                    a = new Return(a.pos, term(env, a[0]));
                    f.blocks.Last().Add(a);
                    f.blocks.Add(new Block("after_return"));
                    return null;
                case ReturnVoid a:
                    f.blocks.Last().Add(a);
                    f.blocks.Add(new Block("after_return"));
                    return null;
                case Print a:
                    {
                        if (a[0] is CompoundExpr u)
                        {
                            for (var i = 0; i < u.Count - 1; i++)
                            {
                                f.blocks.Last().Add(new PrintC(a.pos, term(env, u[i])));
                            }
                            f.blocks.Last().Add(new Print(a.pos, term(env, u.Last())));
                            return null;
                        }
                        f.blocks.Last().Add(new Print(a.pos, term(env, a[0])));
                        return null;
                    }
                case PrintC a:
                    {
                        if (a[0] is CompoundExpr u)
                        {
                            foreach (var b in u)
                            {
                                f.blocks.Last().Add(new PrintC(a.pos, term(env, b)));
                            }
                            return null;
                        }
                        f.blocks.Last().Add(new PrintC(a.pos, term(env, a[0])));
                        return null;
                    }
                case Add a:
                    f.blocks.Last().Add(new Add(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case Sub a:
                    f.blocks.Last().Add(new Sub(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case Mul a:
                    f.blocks.Last().Add(new Mul(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case Div a:
                    f.blocks.Last().Add(new Div(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case Rem a:
                    f.blocks.Last().Add(new Rem(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case Shl a:
                    f.blocks.Last().Add(new Shl(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case Shr a:
                    f.blocks.Last().Add(new Shr(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case BitAnd a:
                    f.blocks.Last().Add(new BitAnd(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case BitOr a:
                    f.blocks.Last().Add(new BitOr(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case BitXor a:
                    f.blocks.Last().Add(new BitXor(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case Eq a:
                    f.blocks.Last().Add(new Eq(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case Le a:
                    f.blocks.Last().Add(new Le(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case Lt a:
                    f.blocks.Last().Add(new Lt(a.pos, term(env, a[0]), term(env, a[1])));
                    return a;
                case BitNot a:
                    f.blocks.Last().Add(new BitNot(a.pos, term(env, a[0])));
                    return a;
                case Neg a:
                    f.blocks.Last().Add(new Neg(a.pos, term(env, a[0])));
                    return a;
                case Literal _:
                    return a0;
                default:
                    throw new ArgumentException(a0.ToString());
            }
        }

        var env1 = new Env(null, null, null);
        foreach (var a in f.body)
            term(env1, a);
        f.blocks.Last().Add(new ReturnVoid(f.pos));

        // resolve labels
        foreach (var block in f.blocks)
        {
            var a0 = block.Last();
            if (a0 is Goto a)
            {
                if (a.target != null)
                    continue;
                if (!f.labels.TryGetValue(a.label, out a.target))
                    throw new Error(a.pos, "label not found");
            }
        }

        // deduplicate names
        HashSet<string> names = new HashSet<string>();

        string uniq(string name)
        {
            if (!names.Contains(name))
            {
                names.Add(name);
                return name;
            }
            for (int i = 1; ; i++)
            {
                var name1 = name + '_' + i;
                if (!names.Contains(name1))
                {
                    names.Add(name1);
                    return name1;
                }
            }
        }

        // deduplicate variable names
        names.Clear();
        foreach (var x in f.vars)
        {
            x.name = uniq(x.name);
        }

        // deduplicate block names
        names.Clear();
        foreach (var block in f.blocks)
        {
            block.name = uniq(block.name);
        }
    }

    static bool pure(Term a)
    {
        switch (a)
        {
            case Literal _:
                return true;
            case Add _:
            case Sub _:
            case Mul _:
            case Div _:
            case Rem _:
            case BitAnd _:
            case BitOr _:
            case BitXor _:
            case BitNot _:
            case Neg _:
            case Eq _:
            case Ne _:
            case Lt _:
            case Le _:
                return a.All(pure);
            default:
                return false;
        }
    }

    public static HashSet<Term> stackOrder(Function f)
    {
        norm(f);
        var stacked = new HashSet<Term>();
        foreach (var block in f.blocks)
        {
            // order all unordered arguments (in particular, literals)
            var unordered = new List<Term>();
            foreach (var a in block)
            {
                foreach (var b in a)
                    if (b.block == null)
                    {
                        b.block = block;
                        unordered.Add(b);
                    }
            }
            block.code.InsertRange(0, unordered);

            // graph for topological sort
            var g = new Graph<Term>(block);

            // arguments must be calculated first
            foreach (var a in block)
                foreach (var user in a.users)
                    if (user.block == block)
                        g.add(a, user);

            // side effects must stay in original order
            var sideEffects = block.Where(a => !pure(a)).ToArray();
            for (var i = 0; i < sideEffects.Length - 1; i++)
            {
                g.add(sideEffects[i], sideEffects[i + 1]);
            }

            // terminator must stay last
            for (var i = 0; i < block.Count - 1; i++)
            {
                g.add(block[i], block.Last());
            }

            // arguments should preferably be calculated in order
            foreach (var a in block)
                a.index = 0;
            int j = 1;

            void visit(Term a)
            {
                if (a.block != block)
                    return;
                if (a.index > 0)
                    return;
                foreach (var b in a)
                    visit(b);
                a.index = j++;
            }

            foreach (var a in block)
                if (!a.users.Any())
                    visit(a);

            // sort
            var sorted = g.topoSort(new IndexCmp());
            if (sorted != null)
                block.code = sorted;

            // which terms can live on the stack
            var stack = new List<Term>();
            foreach (var a in block)
            {
                for (var i = a.Count; i-- > 0;)
                {
                    var b = a[i];
                    if (b.users.Count > 1)
                        goto next;
                    for (; ; )
                    {
                        if (!stack.Any())
                            goto next;
                        if (Etc.pop(stack) == b)
                        {
                            break;
                        }
                    }
                }
                foreach (var b in a)
                    stacked.Add(b);
                next:
                if (AST.getType(a) is TVoid)
                    continue;
                stack.Add(a);
            }
        }
        return stacked;
    }

    class IndexCmp : IComparer<Term>
    {
        int IComparer<Term>.Compare(Term a, Term b)
        {
            if (a.index < b.index)
                return -1;
            if (a.index > b.index)
                return 1;
            return 0;
        }
    }

    public void dump()
    {
        foreach (var f in funcs)
            dump(f);
    }

    public static void dump(Function f)
    {
        norm(f);
        Console.WriteLine("static void {0}() {{", f.name);
        foreach (var x in f.vars)
            Console.WriteLine("int {0};", x);
        foreach (var block in f.blocks)
            dump(block);
        Console.WriteLine('}');
        Console.WriteLine();
    }

    public static void dump(Block block)
    {
        Console.WriteLine(block.name + ':');
        foreach (var a0 in block)
        {
            Console.Write("  ");
            switch (a0)
            {
                case Literal _:
                    Console.WriteLine("{0} = {1};",
                        operand(a0),
                        a0);
                    break;
                case Store a:
                    Console.WriteLine("{0} = {1};",
                        a.x,
                        operand(a[0]));
                    break;
                case Load a:
                    Console.WriteLine("var {0} = {1};",
                        operand(a),
                        a.x);
                    break;
                case Neg a:
                    Console.WriteLine("var {0} = -{1};",
                        operand(a),
                        operand(a[0]));
                    break;
                case BitNot a:
                    Console.WriteLine("var {0} = ~{1};",
                        operand(a),
                        operand(a[0]));
                    break;
                case Add a:
                    Console.WriteLine("var {0} = {1} + {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Sub a:
                    Console.WriteLine("var {0} = {1} - {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Mul a:
                    Console.WriteLine("var {0} = {1} * {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Div a:
                    Console.WriteLine("var {0} = {1} / {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Rem a:
                    Console.WriteLine("var {0} = {1} % {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Shl a:
                    Console.WriteLine("var {0} = {1} << {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Shr a:
                    Console.WriteLine("var {0} = {1} >> {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case BitAnd a:
                    Console.WriteLine("var {0} = {1} & {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case BitOr a:
                    Console.WriteLine("var {0} = {1} | {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case BitXor a:
                    Console.WriteLine("var {0} = {1} ^ {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Lt a:
                    Console.WriteLine("var {0} = {1} < {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Le a:
                    Console.WriteLine("var {0} = {1} <= {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Eq a:
                    Console.WriteLine("var {0} = {1} == {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Ne a:
                    Console.WriteLine("var {0} = {1} != {2};",
                        operand(a),
                        operand(a[0]),
                        operand(a[1]));
                    break;
                case Goto a:
                    Console.WriteLine("goto {0};",
                        a.target != null ? a.target.ToString() : '"' + a.label + '"');
                    break;
                case Br a:
                    Console.WriteLine("if ({0}) goto {1}; else goto {2};",
                        operand(a[0]),
                        a.ifTrue,
                        a.ifFalse);
                    break;
                case Return a:
                    Console.WriteLine("return {0};",
                        operand(a[0]));
                    break;
                case ReturnVoid _:
                    Console.WriteLine("return;");
                    break;
                case Print a:
                    Console.WriteLine("print({0});",
                        operand(a[0]));
                    break;
                case PrintC a:
                    Console.WriteLine("printc({0});",
                        operand(a[0]));
                    break;
                default:
                    throw new ArgumentException(a0.ToString());
            }
        }
    }

    public static void norm(Function f)
    {
        // block
        foreach (var block in f.blocks)
            foreach (var a in block)
                a.block = block;

        // terms
        f.terms.Clear();
        foreach (var block in f.blocks)
            foreach (var a in block)
                f.terms.Add(a);

        // index
        int i = 0;
        foreach (var a in f.terms)
            a.index = i++;

        // users
        foreach (var a in f.terms)
            foreach (var b in a)
                b.users.Clear();
        foreach (var a in f.terms)
            foreach (var b in a)
                b.users.Add(a);
    }

    static string operand(Term a)
    {
        if (a.block == null)
            return a.ToString();
        return "_" + a.index;
    }

    class Env
    {
        public readonly Env parent;
        public readonly Block continueTarget;
        public readonly Block breakTarget;
        public readonly Dictionary<string, Var> vars = new Dictionary<string, Var>();
        public readonly Dictionary<string, Function> funcs = new Dictionary<string, Function>();

        public Env(Env parent, Block continueTarget, Block breakTarget)
        {
            this.parent = parent;
            this.continueTarget = continueTarget;
            this.breakTarget = breakTarget;
        }
    }

    static void put(Env env, string name, Var a)
    {
        env.vars[name] = a;
    }

    static void add(Env env, string name, Function a)
    {
        env.funcs.Add(name, a);
    }

    static Var get(Env env, string name)
    {
        for (; env != null; env = env.parent)
            if (env.vars.TryGetValue(name, out Var a))
                return a;
        return null;
    }

    static Block breakTarget(Env env)
    {
        for (; env != null; env = env.parent)
            if (env.breakTarget != null)
                return env.breakTarget;
        return null;
    }

    static Block continueTarget(Env env)
    {
        for (; env != null; env = env.parent)
            if (env.continueTarget != null)
                return env.continueTarget;
        return null;
    }
}
