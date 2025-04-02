msbuild /p:Configuration=Debug /p:Platform="Any CPU"||exit /b
copy \t\1.cs \t\2.cs||exit /b
bin\Debug\net7.0\clean-cs.exe -i \t\2.cs||exit /b
bat \t\2.cs
