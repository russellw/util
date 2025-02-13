; test.ll
define i32 @test() {
    ; Using a numbered temporary %1
    %1 = add i32 10, 20
    
    ; Using a named value %""
    %"" = add i32 30, 40
    
    ; Using both to show they're different
    %result = add i32 %1, %""
    
    ret i32 %result
}

;empty-name.ll:10:27: error: use of undefined value '%'
;   10 |     %result = add i32 %1, %""
;      |                           ^
;1 error generated.
