using SkiaSharp;

namespace DiagramForge;
public sealed class Text: Window {
	public SKColor color = new(0, 0, 0);
	public string text;
	public float textSize = 12;

	public Text(string text) {
		this.text = text;
	}

	public override void Draw(SKCanvas canvas) {
		using var paint = Paint();
		canvas.DrawText(text, x, y, paint);
		base.Draw(canvas);
	}

	public override void SetSize() {
		using var paint = Paint();
		width = paint.MeasureText(text);
		height = paint.FontSpacing;
	}

	public override string ToString() {
		return text;
	}

	SKPaint Paint() {
		var paint = new SKPaint();
		paint.Color = color;
		paint.TextSize = textSize;
		return paint;
	}
}
