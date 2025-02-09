define i32 @main() {
    %result = add i32 1, mul (i32 2, i32 3)
    ret i32 %result
}
