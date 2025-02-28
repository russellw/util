target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.40.33808"

$__local_stdio_printf_options = comdat any

$"??_C@_05CJBACGMB@hello?$AA@" = comdat any

@"??_C@_05CJBACGMB@hello?$AA@" = linkonce_odr dso_local unnamed_addr constant [6 x i8] c"hello\00", comdat, align 1
@__local_stdio_printf_options._OptionsStorage = internal global i64 0, align 8
