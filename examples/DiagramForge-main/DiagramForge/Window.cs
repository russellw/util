using SkiaSharp;

namespace DiagramForge;
public abstract class Window {
	public List<Window> contents = new();
	public float width = 100, height = 100;
	public float x, y;

	public virtual void Draw(SKCanvas canvas) {
		foreach (var window in contents)
			window.Draw(canvas);
	}

	public virtual void SetPosition(float x, float y) {
		this.x = x;
		this.y = y;
		foreach (var window in contents) {
			window.SetPosition(x, y);
			y += window.height;
		}
	}

	public virtual void SetSize() {
		width = 0;
		height = 0;
		foreach (var window in contents) {
			window.SetSize();
			width = Math.Max(width, window.width);
			height += window.height;
		}
	}
}
