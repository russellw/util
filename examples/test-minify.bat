call html-minifier index.html --collapse-whitespace --minify-css --remove-optional-tags --remove-attribute-quotes -o build\index.html||exit /b
fc \t\1.html build\index.html
copy build\index.html \t\1.html
