import inspect


class Env(dict):
    def __init__(self, outer=None, params=(), args=()):
        self.outer = outer
        self.update(zip(params, args))

    def count(self):
        n = 0
        env = self
        while env:
            n += len(env)
            env = env.outer
        return n

    def get(self, k):
        env = self
        while env:
            if k in env:
                return env[k]
            env = env.outer
        raise ValueError(k)

    def keys1(self):
        s = set()
        env = self
        while env:
            s.update(env.keys())
            env = env.outer
        return s


class Var:
    # this class is intended for logic variables, not necessarily program variables
    def __init__(self, t=None):
        self.t = t

    def __repr__(self):
        if not hasattr(self, "name"):
            return "Var"
        return self.name


def const(a):
    match a:
        case str():
            return
        case Var():
            return
        case ():
            return True
        case "quote", _:
            return True
        case *_,:
            return
    return True


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


def freeVars(a):
    free = set()

    def rec(bound, a):
        match a:
            case Var() as a:
                if a not in bound:
                    free.add(a)
            case "lambda", params, body:
                rec(bound | set(params), body)
            case _, *s:
                for a in s:
                    rec(bound, a)

    rec(set(), a)
    return free


def freshVars(a):
    d = {}

    def rec(a):
        match a:
            case Var() as a:
                if a not in d:
                    d[a] = Var(a.t)
            case _, *s:
                for a in s:
                    rec(a)

    rec(a)
    return replace(d, a)


def replace(d, a):
    if a in d:
        return replace(d, d[a])
    if isinstance(a, tuple):
        return tuple([replace(d, b) for b in a])
    return a


if __name__ == "__main__":
    a = "a"
    x = Var()

    assert const(True)
    assert const(1)
    assert not const(a)
    assert not const(x)
    assert const(())
    assert const(("quote", "a"))
    assert not const(("not", "a"))

    assert freeVars("a") == set()
    assert freeVars(x) == set([x])
    assert freeVars(("+", x, x)) == set([x])

    print("ok")
