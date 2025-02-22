; Module declaration
declare i32 @puts(i8*)

@true_msg = constant [15 x i8] c"This is true!\0A\00"
@false_msg = constant [16 x i8] c"This is false!\0A\00"

define i32 @main() {
    ; Branch directly on constant true
    br i1 true, label %true_block, label %false_block
    
true_block:
    ; Print "This is true!"
    %true_ptr = getelementptr [15 x i8], [15 x i8]* @true_msg, i32 0, i32 0
    call i32 @puts(i8* %true_ptr)
    br label %exit_block
    
false_block:
    ; Print "This is false!"
    %false_ptr = getelementptr [16 x i8], [16 x i8]* @false_msg, i32 0, i32 0
    call i32 @puts(i8* %false_ptr)
    br label %exit_block
    
exit_block:
    ret i32 0
}