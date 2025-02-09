; ModuleID = 'forward_ref_test'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function that demonstrates forward references in phi nodes
define i32 @test_forward_refs(i1 %cond) {
merge:
  ; Using %x and %y before they're defined
  %result = phi i32 [ %x, %then ], [ %y, %else ]
  ret i32 %result

entry:
  br i1 %cond, label %then, label %else

then:
  %x = add i32 42, 10
  br label %merge

else:
  %y = mul i32 21, 2
  br label %merge
}

; External declaration of printf
declare i32 @printf(i8* nocapture readonly, ...)

; String constant for output
@.str = private unnamed_addr constant [18 x i8] c"Result value: %d\0A\00", align 1

; Main function to test our forward_refs function
define i32 @main() {
entry:
  ; Test with condition = true
  %result1 = call i32 @test_forward_refs(i1 1)
  call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 %result1)
  
  ; Test with condition = false
  %result2 = call i32 @test_forward_refs(i1 0)
  call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 %result2)
  
  ret i32 0
}
