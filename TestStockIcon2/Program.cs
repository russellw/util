using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Forms;

public partial class MainForm : Form
{
    [DllImport("shell32.dll", SetLastError = true)]
    public static extern int SHGetStockIconInfo(uint siid, uint uFlags, ref STOCKICONINFO psii);

    public enum StockIconID : uint
    {
        SIID_CUT = 0x0000002f,
        SIID_COPY = 0x00000032,
        SIID_PASTE = 0x0000002d,
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct STOCKICONINFO
    {
        public uint cbSize;
        public IntPtr hIcon;
        public int iSysImageIndex;
        public int iIcon;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
        public string szPath;
    }

    public MainForm()
    {
        InitializeComponent();
    }

    private void InitializeComponent()
    {
        this.SuspendLayout();
        // Initialize and configure form here
        this.ResumeLayout(false);
        LoadIcons();
    }

    private void LoadIcons()
    {
        AddIcon(StockIconID.SIID_CUT, 10, 10);
        AddIcon(StockIconID.SIID_COPY, 75, 10);
        AddIcon(StockIconID.SIID_PASTE, 140, 10);
    }

    private void AddIcon(StockIconID stockIcon, int x, int y)
    {
        STOCKICONINFO sii = new STOCKICONINFO();
        sii.cbSize = (uint)Marshal.SizeOf(typeof(STOCKICONINFO));
        SHGetStockIconInfo((uint)stockIcon, 0x000000100, ref sii);

        PictureBox pb = new PictureBox
        {
            Image = Icon.FromHandle(sii.hIcon).ToBitmap(),
            Location = new Point(x, y),
            Size = new Size(32, 32),
            SizeMode = PictureBoxSizeMode.StretchImage
        };
        this.Controls.Add(pb);
    }

    [STAThread]
    static void Main()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new MainForm());
    }
}
