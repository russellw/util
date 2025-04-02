using SkiaSharp;

namespace DiagramForge;
public sealed class Rectangle: Window {
	public SKColor color = new(0xff, 0xff, 0xff);

	public override void Draw(SKCanvas canvas) {
		using var paint = new SKPaint();
		paint.Color = color;
		canvas.DrawRect(x, y, width, height, paint);
		base.Draw(canvas);
	}
}
