import etc

text = open("/t/a.txt").read()
image = etc.text_to_icon(text)
image.save("a.png")
