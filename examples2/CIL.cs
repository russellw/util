using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;

static class CIL
{
    public static void emitCil(Program program)
    {
        // main class
        var asm = AssemblyBuilder.DefineDynamicAssembly(new AssemblyName("a"), AssemblyBuilderAccess.Save);
        var module = asm.DefineDynamicModule("a", "a.exe", true);
        var t = module.DefineType("Main", TypeAttributes.Public | TypeAttributes.Abstract | TypeAttributes.Sealed);

        // main function
        var main = emit(t, program.funcs[0]);
        asm.SetEntryPoint(main, PEFileKinds.ConsoleApplication);

        // other functions
        for (var i = 1; i < program.funcs.Count; i++)
        {
            emit(t, program.funcs[i]);
        }

        // save
        t.CreateType();
        asm.Save("a.exe");
    }

    static MethodBuilder emit(TypeBuilder t, Function f)
    {
        var stacked = Program.stackOrder(f);
        var m = t.DefineMethod(f.name, MethodAttributes.Public | MethodAttributes.Static);
        var g = m.GetILGenerator();

        // local variables for SSA variables
        var varLocals = new Dictionary<Var, LocalBuilder>();
        foreach (var x in f.vars)
        {
            var local = g.DeclareLocal(typeof(int));
            if (x.name != null)
                local.SetLocalSymInfo(x.name);
            varLocals.Add(x, local);
        }

        // local variables for SSA terms
        var termLocals = new Dictionary<Term, LocalBuilder>();
        foreach (var a in f.terms)
        {
            if (AST.getType(a) is TVoid)
                continue;
            if (a.users.Count == 0)
                continue;
            if (stacked.Contains(a))
                continue;
            var local = g.DeclareLocal(typeof(int));
            local.SetLocalSymInfo("_" + a.index);
            termLocals.Add(a, local);
        }

        // labels
        var labels = new Dictionary<Block, System.Reflection.Emit.Label>();
        foreach (var block in f.blocks)
        {
            labels.Add(block, g.DefineLabel());
        }

        // offsets
        var offsets = new Dictionary<Block, int>();

        bool shortBr(Term a, Block target)
        {
            if (offsets.TryGetValue(target, out var targetOffset))
                return Math.Abs(targetOffset - g.ILOffset) < 128;
            return target[0].index == a.index + 1;
        }

        // code
        void loadInt(int n)
        {
            switch (n)
            {
                case -1:
                    g.Emit(OpCodes.Ldc_I4_M1);
                    break;
                case 0:
                    g.Emit(OpCodes.Ldc_I4_0);
                    break;
                case 1:
                    g.Emit(OpCodes.Ldc_I4_1);
                    break;
                case 2:
                    g.Emit(OpCodes.Ldc_I4_2);
                    break;
                case 3:
                    g.Emit(OpCodes.Ldc_I4_3);
                    break;
                case 4:
                    g.Emit(OpCodes.Ldc_I4_4);
                    break;
                case 5:
                    g.Emit(OpCodes.Ldc_I4_5);
                    break;
                case 6:
                    g.Emit(OpCodes.Ldc_I4_6);
                    break;
                case 7:
                    g.Emit(OpCodes.Ldc_I4_7);
                    break;
                case 8:
                    g.Emit(OpCodes.Ldc_I4_8);
                    break;
                default:
                    if (-128 <= n && n <= 127)
                    {
                        g.Emit(OpCodes.Ldc_I4_S, n);
                        break;
                    }
                    g.Emit(OpCodes.Ldc_I4, n);
                    break;
            }
        }

        void load(LocalVariableInfo x)
        {
            var i = x.LocalIndex;
            switch (i)
            {
                case 0:
                    g.Emit(OpCodes.Ldloc_0);
                    break;
                case 1:
                    g.Emit(OpCodes.Ldloc_1);
                    break;
                case 2:
                    g.Emit(OpCodes.Ldloc_2);
                    break;
                case 3:
                    g.Emit(OpCodes.Ldloc_3);
                    break;
                default:
                    if (-128 <= i && i <= 127)
                    {
                        g.Emit(OpCodes.Ldloc_S, i);
                        break;
                    }
                    g.Emit(OpCodes.Ldloc, i);
                    break;
            }
        }

        void store(LocalVariableInfo x)
        {
            var i = x.LocalIndex;
            switch (i)
            {
                case 0:
                    g.Emit(OpCodes.Stloc_0);
                    break;
                case 1:
                    g.Emit(OpCodes.Stloc_1);
                    break;
                case 2:
                    g.Emit(OpCodes.Stloc_2);
                    break;
                case 3:
                    g.Emit(OpCodes.Stloc_3);
                    break;
                default:
                    if (-128 <= i && i <= 127)
                    {
                        g.Emit(OpCodes.Stloc_S, i);
                        break;
                    }
                    g.Emit(OpCodes.Stloc, i);
                    break;
            }
        }

        void go(Term a, Block target)
        {
            if (target[0].index != a.index + 1)
                g.Emit(shortBr(a, target) ? OpCodes.Br_S : OpCodes.Br, labels[target]);
        }

        foreach (var block in f.blocks)
        {
            g.MarkLabel(labels[block]);
            offsets.Add(block, g.ILOffset);
            for (var i = 0; i < block.Count; i++)
            {
                var a0 = block[i];

                // load inputs to stack
                foreach (var b in a0)
                    if (termLocals.TryGetValue(b, out var x))
                        load(x);

                // optimize compare and branch
                if (i + 1 < block.Count && block[i + 1] is Br br)
                {
                    switch (a0)
                    {
                        case Eq _:
                            // TODO: 'un' variant for floating point?
                            g.Emit(shortBr(br, br.ifTrue) ? OpCodes.Beq_S : OpCodes.Beq, labels[br.ifTrue]);
                            go(br, br.ifFalse);
                            i++;
                            continue;
                        case Ne _:
                            // TODO: 'un' variant for floating point?
                            g.Emit(shortBr(br, br.ifTrue) ? OpCodes.Bne_Un_S : OpCodes.Bne_Un, labels[br.ifTrue]);
                            go(br, br.ifFalse);
                            i++;
                            continue;
                        case Lt _:
                            g.Emit(shortBr(br, br.ifTrue) ? OpCodes.Blt_S : OpCodes.Blt, labels[br.ifTrue]);
                            go(br, br.ifFalse);
                            i++;
                            continue;
                        case Le _:
                            g.Emit(shortBr(br, br.ifTrue) ? OpCodes.Ble_S : OpCodes.Ble, labels[br.ifTrue]);
                            go(br, br.ifFalse);
                            i++;
                            continue;
                    }
                }

                // opcode
                switch (a0)
                {
                    case False _:
                        g.Emit(OpCodes.Ldc_I4_0);
                        break;
                    case True _:
                        g.Emit(OpCodes.Ldc_I4_1);
                        break;
                    case Null _:
                        g.Emit(OpCodes.Ldnull);
                        break;
                    case CharLiteral a:
                        loadInt(a.val);
                        break;
                    case IntLiteral a:
                        loadInt(a.val);
                        break;
                    case StringLiteral a:
                        g.Emit(OpCodes.Ldstr, a.val);
                        break;
                    case Load a:
                        load(varLocals[a.x]);
                        break;
                    case Store a:
                        store(varLocals[a.x]);
                        break;
                    case Print a:
                        {
                            var mi = typeof(Console).GetMethod("WriteLine", new[] { cilType(a0[0]) });
                            g.Emit(OpCodes.Call, mi);
                            break;
                        }
                    case PrintC a:
                        {
                            var mi = typeof(Console).GetMethod("Write", new[] { cilType(a0[0]) });
                            g.Emit(OpCodes.Call, mi);
                            break;
                        }
                    case Neg a:
                        g.Emit(OpCodes.Neg);
                        break;
                    case BitNot a:
                        g.Emit(OpCodes.Not);
                        break;
                    case Add a:
                        g.Emit(OpCodes.Add);
                        break;
                    case Sub a:
                        g.Emit(OpCodes.Sub);
                        break;
                    case Mul a:
                        g.Emit(OpCodes.Mul);
                        break;
                    case Div a:
                        g.Emit(OpCodes.Div);
                        break;
                    case Rem a:
                        g.Emit(OpCodes.Rem);
                        break;
                    case Shl a:
                        g.Emit(OpCodes.Shl);
                        break;
                    case Shr a:
                        g.Emit(OpCodes.Shr);
                        break;
                    case BitAnd a:
                        g.Emit(OpCodes.And);
                        break;
                    case BitOr a:
                        g.Emit(OpCodes.Or);
                        break;
                    case BitXor a:
                        g.Emit(OpCodes.Xor);
                        break;
                    case Eq _:
                        g.Emit(OpCodes.Ceq);
                        break;
                    case Ne _:
                        g.Emit(OpCodes.Ceq);
                        g.Emit(OpCodes.Ldc_I4_0);
                        g.Emit(OpCodes.Ceq);
                        break;
                    case Lt _:
                        // TODO: 'un' variant for unsigned numbers
                        g.Emit(OpCodes.Clt);
                        break;
                    case Le _:
                        // TODO: 'un' variant for unsigned numbers and floating point
                        g.Emit(OpCodes.Cgt);
                        g.Emit(OpCodes.Ldc_I4_0);
                        g.Emit(OpCodes.Ceq);
                        break;
                    case ReturnVoid _:
                        g.Emit(OpCodes.Ret);
                        break;
                    case Goto a:
                        go(a, a.target);
                        break;
                    case Br a:
                        g.Emit(shortBr(a, a.ifTrue) ? OpCodes.Brtrue_S : OpCodes.Brtrue, labels[a.ifTrue]);
                        go(a, a.ifFalse);
                        break;
                    default:
                        throw new ArgumentException(a0.ToString());
                }

                // store output from stack
                if (AST.getType(a0) is TVoid)
                    continue;
                if (stacked.Contains(a0))
                    continue;
                if (termLocals.TryGetValue(a0, out var x1))
                {
                    store(x1);
                    continue;
                }
                g.Emit(OpCodes.Pop);
            }
        }
        return m;
    }

    static Type cilType(Term a)
    {
        var type = AST.getType(a);
        switch (type)
        {
            case TString _:
                return typeof(string);
            case TInt _:
                return typeof(int);
            case TBool _:
                return typeof(bool);
            case TChar _:
                return typeof(char);
            default:
                throw new ArgumentException(a + " " + type);
        }
    }
}
