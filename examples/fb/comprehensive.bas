'' ============================================================
'' FreeBASIC Language Features Demonstration
'' ============================================================
'' This program demonstrates various features of the FreeBASIC language
'' including data types, control structures, procedures, custom types,
'' graphics, and more.

#include "fbgfx.bi"  '' Include graphics library

'' ============================================================
'' Type Definitions
'' ============================================================
Type Person
    As String Name
    As Integer Age
    As Single Height
    As Boolean IsEmployed
End Type

Type Vector3D
    As Single x, y, z
    
    Declare Function Magnitude() As Single
    Declare Sub Normalize()
    Declare Function ToString() As String
End Type

'' Method implementation for Vector3D type
Function Vector3D.Magnitude() As Single
    Return Sqr(x * x + y * y + z * z)
End Function

Sub Vector3D.Normalize()
    Dim As Single mag = This.Magnitude()
    If mag > 0 Then
        x /= mag
        y /= mag
        z /= mag
    End If
End Sub

Function Vector3D.ToString() As String
    Return "(" & x & ", " & y & ", " & z & ")"
End Function

'' ============================================================
'' Function and Procedure Declarations
'' ============================================================
Declare Function CalculateFactorial(n As Integer) As ULongInt
Declare Sub DisplayArray(arr() As Integer)
Declare Function IsPrime(n As Integer) As Boolean
Declare Sub DrawFractal(x As Integer, y As Integer, size As Integer, depth As Integer)

'' ============================================================
'' Constants
'' ============================================================
Const PI As Double = 3.14159265358979
Const APP_NAME As String = "FreeBASIC Features Demo"
Const MAX_ARRAY_SIZE As Integer = 10
Const TRUE As Integer = -1
Const FALSE As Integer = 0

'' ============================================================
'' Main Program
'' ============================================================
Randomize Timer '' Initialize random number generator

'' Screen setup with error handling
Dim As Integer screenWidth = 800, screenHeight = 600
Screen 0
On Error Goto GraphicsError
ScreenRes screenWidth, screenHeight, 32
On Error Goto 0

'' Variable declarations showing different data types
Dim As Byte byteVar = 255
Dim As UByte ubyteVar = 255
Dim As Short shortVar = -32768
Dim As UShort ushortVar = 65535
Dim As Integer intVar = -2147483648
Dim As UInteger uintVar = 4294967295
Dim As LongInt longVar = -9223372036854775808LL
Dim As ULongInt ulongVar = 18446744073709551615ULL
Dim As Single singleVar = 3.14159
Dim As Double doubleVar = 2.71828182845904
Dim As String strVar = "Hello, FreeBASIC!"
Dim As ZString * 20 zstrVar = "Fixed-length string"
Dim As WString * 20 wstrVar = WStr("Wide character string")
Dim As Boolean boolVar = TRUE

'' Array declaration and initialization
Dim As Integer numbers(1 To MAX_ARRAY_SIZE)
Dim As String names(5)
names(0) = "Alice"
names(1) = "Bob"
names(2) = "Charlie"
names(3) = "David"
names(4) = "Eve"
names(5) = "Frank"
Dim As Double matrix(3, 3)  '' 2D array

'' Dynamic array
Dim As Single Ptr dynamicArray
dynamicArray = Allocate(5 * SizeOf(Single))
For i As Integer = 0 To 4
    dynamicArray[i] = Rnd() * 100
Next
    
'' Initialize the array with random values
For i As Integer = 1 To MAX_ARRAY_SIZE
    numbers(i) = Int(Rnd() * 100)
Next

'' Initialize matrix with values
For row As Integer = 0 To 2
    For col As Integer = 0 To 2
        matrix(row, col) = row * 3 + col + 1
    Next
Next

'' Create a Person object
Dim As Person someone
someone.Name = "John Doe"
someone.Age = 35
someone.Height = 1.85
someone.IsEmployed = TRUE

'' Create a Vector3D object
Dim As Vector3D v
v.x = 3.0
v.y = 4.0
v.z = 5.0

'' ============================================================
'' Demonstration of Control Structures
'' ============================================================

'' If-Then-Else demonstration
Print "=== Conditional Statements ==="
If someone.Age < 18 Then
    Print someone.Name & " is under 18 years old."
ElseIf someone.Age < 65 Then
    Print someone.Name & " is an adult of working age."
Else
    Print someone.Name & " is a senior citizen."
End If

'' Select Case demonstration
Print
Print "=== Select Case Statement ==="
Select Case someone.Age
    Case Is < 13
        Print "Child"
    Case 13 To 19
        Print "Teenager"
    Case 20 To 64
        Print "Adult"
    Case Else
        Print "Senior"
End Select

'' For loop demonstration
Print
Print "=== For Loop ==="
Print "First 5 Factorials:"
For i As Integer = 1 To 5
    Print "Factorial of " & i & " is " & CalculateFactorial(i)
Next

'' While loop demonstration
Print
Print "=== While Loop ==="
Print "Prime numbers less than 30:"
Dim As Integer num = 2
While num < 30
    If IsPrime(num) Then Print num;
    num += 1
Wend
Print

'' Do loop demonstration
Print
Print "=== Do Loop ==="
Print "Countdown:"
Dim As Integer countdown = 5
Do
    Print countdown;
    countdown -= 1
    Sleep 100
Loop Until countdown < 0
Print " Blast off!"

'' Iterating through array manually (instead of For Each)
Print
Print "=== Array Iteration ==="
Print "Names list:"
For i As Integer = 0 To 5
    Print names(i);
Next
Print

'' ============================================================
'' Advanced Features Demonstration
'' ============================================================

'' Using custom type methods
Print
Print "=== Custom Types with Methods ==="
Print "Vector: " & v.ToString()
Print "Magnitude: " & v.Magnitude()
v.Normalize()
Print "After normalization: " & v.ToString()

'' Pointer demonstration
Print
Print "=== Pointers ==="
Print "Dynamic array values:"
For i As Integer = 0 To 4
    Print dynamicArray[i];
Next
Print
Deallocate(dynamicArray)

'' Array handling with procedure
Print
Print "=== Array Handling ==="
Print "Numbers array:"
DisplayArray(numbers())

'' ============================================================
'' Graphics Demonstration
'' ============================================================
Print
Print "=== Graphics Demonstration ==="
Print "Press any key to continue to graphics demo..."
Sleep
Cls

'' Draw some basic shapes
Line (0, 0)-(screenWidth-1, screenHeight-1), RGB(255, 0, 0), B
Circle (screenWidth\2, screenHeight\2), 100, RGB(0, 255, 0)
Paint (screenWidth\2, screenHeight\2), RGB(0, 128, 0), RGB(0, 255, 0)

'' Draw text on screen
Draw String (10, 10), "FreeBASIC Graphics Demo", RGB(255, 255, 255)
Draw String (10, 30), "Press ESC to exit", RGB(255, 255, 255)

'' Draw a simple fractal
DrawFractal(screenWidth\2, screenHeight\2, 150, 5)

'' Wait for key press
Dim As String keypress
While keypress <> Chr(27)  '' ESC key
    keypress = Inkey()
    Sleep 10
Wend

Screen 0  '' Return to text mode
Print "Graphics demonstration complete."

'' Program end
Print
Print "=== Program Complete ==="
Print "This concludes the FreeBASIC language features demonstration."
Print "Press any key to exit..."
Sleep
End 0

'' Skip to here if graphics initialization fails
GraphicsError:
Print "Graphics initialization failed. Running in text mode only."
Goto ContinueWithoutGraphics

ContinueWithoutGraphics:

'' ============================================================
'' Function and Procedure Implementations
'' ============================================================

'' Calculate factorial
Function CalculateFactorial(n As Integer) As ULongInt
    If n <= 1 Then Return 1
    Return n * CalculateFactorial(n - 1)
End Function

'' Display array contents
Sub DisplayArray(arr() As Integer)
    Dim As Integer lowerBound = LBound(arr)
    Dim As Integer upperBound = UBound(arr)
    
    For i As Integer = lowerBound To upperBound
        Print arr(i);
    Next
    Print
End Sub

'' Check if a number is prime
Function IsPrime(n As Integer) As Boolean
    If n <= 1 Then Return FALSE
    If n <= 3 Then Return TRUE
    If (n Mod 2 = 0) Or (n Mod 3 = 0) Then Return FALSE
    
    Dim As Integer i = 5
    While i * i <= n
        If (n Mod i = 0) Or (n Mod (i + 2) = 0) Then Return FALSE
        i += 6
    Wend
    
    Return TRUE
End Function

'' Recursive function to draw a fractal
Sub DrawFractal(x As Integer, y As Integer, size As Integer, depth As Integer)
    If depth <= 0 Then Exit Sub
    
    '' Draw a square
    Line (x - size\2, y - size\2)-(x + size\2, y + size\2), RGB(0, 0, 255), B
    
    '' Recursively draw smaller squares at the corners
    Dim As Integer newSize = size \ 2
    If depth > 1 Then
        DrawFractal(x - newSize, y - newSize, newSize, depth - 1)
        DrawFractal(x + newSize, y - newSize, newSize, depth - 1)
        DrawFractal(x - newSize, y + newSize, newSize, depth - 1)
        DrawFractal(x + newSize, y + newSize, newSize, depth - 1)
    End If
End Sub