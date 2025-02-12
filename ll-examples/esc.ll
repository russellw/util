; Test case for LLVM string escape sequences
; Define a helper function to print strings
declare i32 @puts(ptr nocapture) nounwind

@str1 = constant [3 x i8] c"\41\0A\00"   
@str2 = constant [3 x i8] c"\\\0A\00"   
@str4 = constant [4 x i8] c"\?\0A\00"   
@str5 = constant [4 x i8] c"\'\0A\00"   
@str6 = constant [4 x i8] c"\n\0A\00"   
@str7 = constant [4 x i8] c"\t\0A\00"   
@str8 = constant [4 x i8] c"\r\0A\00"   
@str9 = constant [4 x i8] c"\f\0A\00"   
@str10 = constant [4 x i8] c"\v\0A\00"   
@str11 = constant [4 x i8] c"\e\0A\00"   

define i32 @main() {
    ; Print all test strings
    call i32 @puts(ptr @str1)
    call i32 @puts(ptr @str2)
    call i32 @puts(ptr @str4)
    call i32 @puts(ptr @str5)
    call i32 @puts(ptr @str6)
    call i32 @puts(ptr @str7)
    call i32 @puts(ptr @str8)
    call i32 @puts(ptr @str9)
    call i32 @puts(ptr @str10)
    call i32 @puts(ptr @str11)

    ret i32 0
}
