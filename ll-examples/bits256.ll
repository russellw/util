; ModuleID = 'i256_test'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@format = private constant [21 x i8] c"Upper 64 bits: %llx\0A\00"

declare i32 @printf(i8* nocapture readonly, ...) 

define i256 @add_i256(i256 %a, i256 %b) {
entry:
    %sum = add i256 %a, %b
    ret i256 %sum
}

define i256 @mul_i256(i256 %a, i256 %b) {
entry:
    %product = mul i256 %a, %b
    ret i256 %product
}

define i32 @main() {
entry:
    ; Create some 256-bit values
    ; 2^255 + 1
    %a = add i256 1, shl i256 1, 255
    ; 2^128 + 1
    %b = add i256 1, shl i256 1, 128
    
    ; Add them
    %sum = call i256 @add_i256(i256 %a, i256 %b)
    
    ; Extract upper 64 bits for printing
    %shifted = lshr i256 %sum, 192
    %upper64 = trunc i256 %shifted to i64
    
    ; Print result
    %printf = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @format, i64 0, i64 0), i64 %upper64)
    
    ; Multiply them
    %product = call i256 @mul_i256(i256 %a, i256 %b)
    
    ; Extract upper 64 bits of product
    %shifted2 = lshr i256 %product, 192
    %upper64_2 = trunc i256 %shifted2 to i64
    
    ; Print result
    %printf2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @format, i64 0, i64 0), i64 %upper64_2)
    
    ret i32 0
}