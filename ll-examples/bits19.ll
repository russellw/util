; ModuleID = 'i19_test'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@format = private constant [12 x i8] c"Result: %d\0A\00"

declare i32 @printf(i8* nocapture readonly, ...) 

define i19 @add_i19(i19 %a, i19 %b) {
entry:
    %sum = add i19 %a, %b
    ret i19 %sum
}

define i19 @mul_i19(i19 %a, i19 %b) {
entry:
    %product = mul i19 %a, %b
    ret i19 %product
}

define i32 @main() {
entry:
    ; Create some 19-bit values
    %a = add i19 0, 500000
    %b = add i19 0, 50000
    
    ; Add them
    %sum = call i19 @add_i19(i19 %a, i19 %b)
    
    ; Convert to i32 for printf
    %sum.ext = zext i19 %sum to i32
    
    ; Print result
    %printf = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @format, i64 0, i64 0), i32 %sum.ext)
    
    ; Multiply them
    %product = call i19 @mul_i19(i19 %a, i19 %b)
    
    ; Convert to i32 for printf
    %product.ext = zext i19 %product to i32
    
    ; Print result
    %printf2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @format, i64 0, i64 0), i32 %product.ext)
    
    ret i32 0
}