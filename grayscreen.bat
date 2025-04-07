cd \t||exit /b
nircmd savescreenshot 1.png

rem Crop other stuff from desktop
magick 1.png -crop +230+0 +repage 2.png||exit /b
magick 2.png -crop +0-48 +repage 3.png||exit /b

rem Replace gray background and border with transparent
magick 3.png -fill #00000000 -floodfill +0+0 #808080 4.png||exit /b

rem Trim the transparent surroundings
magick 4.png -trim +repage 5.png||exit /b

rem Crop the chrome and rounded corners
magick 5.png -crop +0-22 +repage 6.png||exit /b
magick 6.png -crop +0+48 +repage 7.png||exit /b

rem Check the dimensions of the result
magick identify 7.png
