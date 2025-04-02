# TestSHGetFileInfo
Test SHGetFileInfo

```
TestSHGetFileInfo

Installing:

Microsoft.Win32.SystemEvents.8.0.0
System.Drawing.Common.8.0.4


TestSHGetFileInfo

Installing:

WindowsAPICodePack-Core.1.1.1
WindowsAPICodePack-Shell.1.1.1
```

WindowsAPICodePack-Shell has several similarly named packages whose descriptions appear to claim that they are newer versions. However, this is the one with the most downloads.

```
Unhandled exception. System.TypeLoadException: Could not load type 'System.Windows.Size' from assembly 'WindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35'.
   at Program.GetLargeThumbnail(String path)
   at Program.Main(String[] args) in C:\w\TestSHGetFileInfo\Program.cs:line 19
```

The error message indicates that there's an issue with loading a type from the `WindowsBase` assembly, which suggests a potential compatibility issue or a missing reference. This is often encountered when trying to use certain parts of the Windows API Code Pack or other libraries that depend on Windows Presentation Foundation (WPF) components within a console application.

To resolve this issue, you need to ensure that your project references the necessary assemblies. Here are a few steps to troubleshoot and fix this:

### 1. Ensure Appropriate Project Type
Ensure that your project type supports WPF if you are utilizing components that depend on `WindowsBase`. In a typical console application, these components may not be readily available.

### 2. Add Necessary References
If you're committed to using a console application, you need to add references manually:

- **Reference `WindowsBase` Manually**: You can do this in Visual Studio by right-clicking on your project, selecting "Add" -> "Reference...", then browsing for `WindowsBase`.

### 3. Consider Application Type
If your application heavily relies on components from WPF (like handling images in ways specific to WPF), consider whether a WPF application might be more appropriate than a console application.

### Alternative Approach without WPF Dependency
If adding the reference does not resolve the issue or if you prefer not to use WPF components, consider using a simpler method that does not rely on the Windows API Code Pack. Instead, you can use `System.Drawing` directly, along with basic PInvoke, to fetch and save the icon. Here’s how you can modify the code to avoid WPF dependencies:

```csharp
using System;
using System.Drawing;
using System.Runtime.InteropServices;

class Program
{
    [DllImport("shell32.dll", CharSet = CharSet.Auto)]
    public static extern IntPtr SHGetFileInfo(string pszPath, uint dwFileAttributes, ref SHFILEINFO psfi, uint cbSizeFileInfo, uint uFlags);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public struct SHFILEINFO
    {
        public IntPtr hIcon;
        public int iIcon;
        public uint dwAttributes;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
        public string szDisplayName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 80)]
        public string szTypeName;
    }

    private const uint SHGFI_ICON = 0x100;
    private const uint SHGFI_LARGEICON = 0x0;

    public static Icon GetFileIcon(string path)
    {
        SHFILEINFO shfi = new SHFILEINFO();
        IntPtr hIcon = SHGetFileInfo(path, 0, ref shfi, (uint)Marshal.SizeOf(shfi), SHGFI_ICON | SHGFI_LARGEICON);
        if (hIcon != IntPtr.Zero)
        {
            return Icon.FromHandle(shfi.hIcon);
        }
        return null;
    }

    static void Main()
    {
        string filePath = @"C:\Path\To\FileOrExecutable.exe"; // Adjust the path to your file
        string savePath = @"C:\Path\To\Save\Icon.png"; // Adjust the save path

        Icon fileIcon = GetFileIcon(filePath);
        if (fileIcon != null)
        {
            using (Bitmap iconBitmap = fileIcon.ToBitmap())
            {
                iconBitmap.Save(savePath, System.Drawing.Imaging.ImageFormat.Png);
                Console.WriteLine($"Icon saved to {savePath}");
            }
        }
        else
        {
            Console.WriteLine("Failed to retrieve the icon.");
        }
    }
}
```

This code doesn't attempt to fetch a 256x256 icon because the SHGetFileInfo without additional configuration (like linking with the system image list) will not retrieve icons at this size but should work for standard sizes like 32x32 or 48x48 depending on system settings.
