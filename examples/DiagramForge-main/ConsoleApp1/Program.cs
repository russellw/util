using SkiaSharp;

class Program {
	static void Main(string[] _) {
		var width = 2;
		var height = 100;
		using var bitmap = new SKBitmap(width, height);
		using var canvas = new SKCanvas(bitmap);
		string[] parts = { "int", " ", "x" };
		using var paint = new SKPaint { TextSize = 12, Color = SKColors.Black };
		var position = new SKPoint(10, 10);
		SKColor[] colors = { SKColors.Blue, SKColors.Black, SKColors.White };
		for (var i = 0; i < parts.Length; i++) {
			paint.Color = colors[i];
			canvas.DrawText(parts[i], position.X, position.Y, paint);
			position.X += paint.MeasureText(parts[i]);
		}
		// Console.WriteLine(position);

		Console.Write("FontSpacing\t");
		Console.WriteLine(paint.FontSpacing);
		Console.WriteLine();

		Console.Write("Leading\t\t");
		Console.WriteLine(paint.FontMetrics.Leading);

		Console.Write("Ascent\t\t");
		Console.WriteLine(paint.FontMetrics.Ascent);

		Console.Write("Bottom\t\t");
		Console.WriteLine(paint.FontMetrics.Bottom);

		Console.Write("CapHeight\t");
		Console.WriteLine(paint.FontMetrics.CapHeight);

		Console.Write("Descent\t\t");
		Console.WriteLine(paint.FontMetrics.Descent);

		Console.Write("Top\t\t");
		Console.WriteLine(paint.FontMetrics.Top);
		Console.WriteLine();

		var fontMetrics = paint.FontMetrics;
		var totalLineHeight = fontMetrics.Descent - fontMetrics.Ascent + fontMetrics.Leading;
		Console.WriteLine(totalLineHeight);
	}
}
