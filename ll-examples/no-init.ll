; Module-level declarations
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Global variable with external linkage (no initializer)
@global_var = external global i32

; Global variable with an initializer (for comparison)
@global_with_init = global i32 42

; Function that uses the global variable
define i32 @get_global() {
    %val = load i32, i32* @global_var
    ret i32 %val
}

; Function that sets the global variable
define void @set_global(i32 %new_val) {
    store i32 %new_val, i32* @global_var
    ret void
}
