; Module declaration
source_filename = "compare_constants.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Global constants
@const1 = constant i32 42
@const2 = constant i32 100

; Constant expressions for different comparison types
@eq_result = constant i1 icmp eq i32 42, 100    ; Equal
@ne_result = constant i1 icmp ne i32 42, 100    ; Not equal
@slt_result = constant i1 icmp slt i32 42, 100  ; Signed less than
@sgt_result = constant i1 icmp sgt i32 42, 100  ; Signed greater than
@sle_result = constant i1 icmp sle i32 42, 100  ; Signed less or equal
@sge_result = constant i1 icmp sge i32 42, 100  ; Signed greater or equal

; Using globals for comparison
@global_eq = constant i1 icmp eq i32 ptrtoint (i32* @const1 to i32), ptrtoint (i32* @const2 to i32)

; Main function demonstrating usage
define i32 @main() {
entry:
    ; Load comparison results
    %1 = load i1, i1* @eq_result
    %2 = load i1, i1* @ne_result
    %3 = load i1, i1* @slt_result
    %4 = load i1, i1* @sgt_result
    %5 = load i1, i1* @sle_result
    %6 = load i1, i1* @sge_result
    %7 = load i1, i1* @global_eq
    
    ; Convert boolean to i32 for return
    %8 = zext i1 %1 to i32
    ret i32 %8
}