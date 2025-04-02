namespace DiagramForge;
public sealed class Row: Window {
	public override void SetPosition(float x, float y) {
		this.x = x;
		this.y = y;
		foreach (var window in contents) {
			window.SetPosition(x, y);
			x += window.width;
		}
	}

	public override void SetSize() {
		width = 0;
		height = 0;
		foreach (var window in contents) {
			window.SetSize();
			width += window.width;
			height = Math.Max(height, window.height);
		}
	}
}
