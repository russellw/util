; Module declaration
source_filename = "complex_constant.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Global constant with complex nested expression
@complex_constant = constant i64 add (
    i64 mul (
        i64 add (
            i64 mul (
                i64 sub (
                    i64 mul (
                        i64 123456789,
                        i64 2
                    ),
                    i64 sub (
                        i64 mul (
                            i64 42,
                            i64 1000000
                        ),
                        i64 33333
                    )
                ),
                i64 16
            ),
            i64 mul (
                i64 mul (
                    i64 7777777,
                    i64 8888888
                ),
                i64 12345
            )
        ),
        i64 add (
            i64 mul (
                i64 mul (
                    i64 555555,
                    i64 666666
                ),
                i64 8
            ),
            i64 mul (
                i64 sub (
                    i64 mul (
                        i64 111111,
                        i64 3
                    ),
                    i64 999999
                ),
                i64 add (
                    i64 12345,
                    i64 67890
                )
            )
        )
    ),
    i64 sub (
        i64 mul (
            i64 mul (
                i64 13579,
                i64 24680
            ),
            i64 456
        ),
        i64 mul (
            i64 add (
                i64 9876,
                i64 5432
            ),
            i64 sub (
                i64 1111,
                i64 2222
            )
        )
    )
)

; Function to return the constant
define i64 @get_complex_constant() {
    ret i64 ptrtoint (i64* @complex_constant to i64)
}