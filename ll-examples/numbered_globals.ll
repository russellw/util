; ModuleID = 'numbered_globals'

; Global variable with number instead of name
@0 = global i32 42

; Function with number instead of name
define i32 @1() {
    %1 = load i32, ptr @0
    ret i32 %1
}

; Main function that calls the numbered function
define i32 @main() {
    %1 = call i32 @1()
    ret i32 %1
}