; ModuleID = 'i512_test'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@format = private constant [21 x i8] c"Upper 64 bits: %llx\0A\00"

declare i32 @printf(i8* nocapture readonly, ...) 

define i512 @add_i512(i512 %a, i512 %b) {
entry:
    %sum = add i512 %a, %b
    ret i512 %sum
}

define i512 @mul_i512(i512 %a, i512 %b) {
entry:
    %product = mul i512 %a, %b
    ret i512 %product
}

define i32 @main() {
entry:
    ; Create some 512-bit values
    %one = add i512 0, 1
    %shift_amt = add i512 0, 511
    %tmp = shl i512 %one, %shift_amt
    %a = add i512 %tmp, 1  ; 2^511 + 1
    
    ; Create second value (2^256 + 1)
    %shift_amt2 = add i512 0, 256
    %tmp2 = shl i512 %one, %shift_amt2
    %b = add i512 %tmp2, 1
    
    ; Add them
    %sum = call i512 @add_i512(i512 %a, i512 %b)
    
    ; Extract upper 64 bits for printing
    %shift_print = add i512 0, 448
    %shifted = lshr i512 %sum, %shift_print
    %upper64 = trunc i512 %shifted to i64
    
    ; Print result
    %printf = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @format, i64 0, i64 0), i64 %upper64)
    
    ; Multiply them
    %product = call i512 @mul_i512(i512 %a, i512 %b)
    
    ; Extract upper 64 bits of product
    %shifted2 = lshr i512 %product, %shift_print
    %upper64_2 = trunc i512 %shifted2 to i64
    
    ; Print result
    %printf2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @format, i64 0, i64 0), i64 %upper64_2)
    
    ret i32 0
}