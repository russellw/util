namespace DiagramForge;
public sealed class Border: Window {
	public float radius = 2;

	public override void SetPosition(float x, float y) {
		this.x = x;
		this.y = y;
		x += radius;
		y += radius;
		foreach (var window in contents) {
			window.SetPosition(x, y);
			y += window.height;
		}
	}

	public override void SetSize() {
		width = 0;
		height = 0;
		foreach (var window in contents) {
			window.SetSize();
			width = Math.Max(width, window.width);
			height += window.height;
		}
		width += radius * 2;
		height += radius * 2;
	}
}
