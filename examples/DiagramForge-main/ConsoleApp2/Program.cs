using DiagramForge;
using SkiaSharp;

class Program {
	static void Main(string[] _) {
		var text = new Text("foo");
		text.y = 500;
		text.textSize = 100;

		using var bitmap = new SKBitmap(1800, 900);
		using var canvas = new SKCanvas(bitmap);
		canvas.Clear(new SKColor(0xff, 0xff, 0xff));
		text.Draw(canvas);

		using var image = SKImage.FromBitmap(bitmap);
		using var data = image.Encode(SKEncodedImageFormat.Png, 100);
		using var stream = new FileStream("bin/a.png", FileMode.Create);
		data.SaveTo(stream);
	}
}
