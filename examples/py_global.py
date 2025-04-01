# https://stackoverflow.com/questions/33211272/what-is-the-difference-between-non-local-variable-and-global-variable

foo = 0 # <- 〇
def outer():
    foo = 5 # <- ✖
    def middle():
        foo = 10 # <- ✖
        def inner():
            global foo # Here
            #foo += 1
            print(foo) # 1
        inner()
    middle()
outer()
