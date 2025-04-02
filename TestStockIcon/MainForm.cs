using Microsoft.WindowsAPICodePack.Shell;
using System.Drawing;

namespace TestStockIcon {
public sealed class MainForm: Form {
	private PictureBox pictureBoxCut;
	private PictureBox pictureBoxCopy;
	private PictureBox pictureBoxPaste;

	public MainForm() {
		InitializeComponent();
		LoadIcons();
	}

	private void InitializeComponent() {
		this.Text = "Icon Example";
		this.ClientSize = new Size(300, 100);

		// Initialize PictureBox for Cut
		pictureBoxCut = new PictureBox { Size = new Size(32, 32), Location = new Point(10, 30), BorderStyle = BorderStyle.Fixed3D };
		this.Controls.Add(pictureBoxCut);

		// Initialize PictureBox for Copy
		pictureBoxCopy =
			new PictureBox { Size = new Size(32, 32), Location = new Point(50, 30), BorderStyle = BorderStyle.Fixed3D };
		this.Controls.Add(pictureBoxCopy);

		// Initialize PictureBox for Paste
		pictureBoxPaste =
			new PictureBox { Size = new Size(32, 32), Location = new Point(90, 30), BorderStyle = BorderStyle.Fixed3D };
		this.Controls.Add(pictureBoxPaste);
	}

	private void LoadIcons() {
		try {
			pictureBoxCut.Image = ShellObject.GetStockIcon(StockIconId.Cut, StockIconOptions.LinkOverlay, 32).Bitmap;
			pictureBoxCopy.Image = ShellObject.GetStockIcon(StockIconId.Copy, StockIconOptions.LinkOverlay, 32).Bitmap;
			pictureBoxPaste.Image = ShellObject.GetStockIcon(StockIconId.Paste, StockIconOptions.LinkOverlay, 32).Bitmap;
		} catch (Exception ex) {
			MessageBox.Show("Error loading icons: " + ex.Message);
		}
	}
}
}