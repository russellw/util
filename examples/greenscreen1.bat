cd \t||exit /b
nircmd savescreenshot 1.png
magick 1.png -crop +230+0 +repage 2.png||exit /b
magick 2.png -crop +0-48 +repage 3.png||exit /b
greenscreen 3.png 4.png||exit /b
magick 4.png -trim +repage 5.png||exit /b
magick identify 5.png
