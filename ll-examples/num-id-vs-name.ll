; test.ll
define i32 @test() {
    ; Using a numbered temporary %1
    %1 = add i32 10, 20
    
    ; Using a named value %"1"
    %"1" = add i32 30, 40
    
    ; Using both to show they're different
    %result = add i32 %1, %"1"
    
    ret i32 %result
}
