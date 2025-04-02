# TestExtractAssociatedIcon
Extract icon associated with a file

```
TestExtractAssociatedIcon

Installing:

Microsoft.Win32.SystemEvents.8.0.0
System.Drawing.Common.8.0.4
```

C:\Users\User\.nuget\packages\system.drawing.common\9.0.0-preview.3.24172.9\buildTransitive\netcoreapp2.0\System.Drawing.Common.targets(4,5): warning : System.Drawing.Common 9.0.0-preview.3.24172.9 doesn't support net7.0 and has not been tested with it. Consider upgrading your TargetFramework to net8.0 or later. You may also set <SuppressTfmSupportBuildWarnings>true</SuppressTfmSupportBuildWarnings> in the project file to ignore this warning and attempt to run in this unsupported configuration at your own risk. [C:\w\TestExtractAssociatedIcon\TestExtractAssociatedIcon.csproj]

An error occurred: System.Drawing.Common is not supported on this platform.

Having installed the correct version, this is the conclusion:

To retrieve a larger icon using the `Icon.ExtractAssociatedIcon` method in C#, you will encounter limitations because this method typically returns the system's default small icon (usually 32x32). For getting larger icons directly, you need to use more detailed API calls like `SHGetFileInfo` with specific flags, or utilize other methods that involve interacting with the system image list.

Here's an enhanced approach using `SHGetFileInfo` to get a larger icon (e.g., 256x256), adapted for your needs. This method is more flexible because it allows you to specify the size of the icon you want to retrieve:

### Using SHGetFileInfo to Extract Larger Icons

This code snippet will help you fetch and save a larger icon, adjusting the size by using the right combination of flags:

```csharp
using System;
using System.Drawing;
using System.Runtime.InteropServices;

class Program
{
    [DllImport("shell32.dll", CharSet = CharSet.Auto)]
    static extern IntPtr SHGetFileInfo(string pszPath, uint dwFileAttributes, ref SHFILEINFO psfi, uint cbSizeFileInfo, uint uFlags);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public struct SHFILEINFO
    {
        public IntPtr hIcon;
        public IntPtr iIcon;
        public uint dwAttributes;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
        public string szDisplayName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 80)]
        public string szTypeName;
    }

    private const uint SHGFI_ICON = 0x100;
    private const uint SHGFI_LARGEICON = 0x0;  // 'Large icon
    private const uint SHGFI_USEFILEATTRIBUTES = 0x10;

    static void Main(string[] args)
    {
        string filePath = args[0];  // Path provided via command line argument
        string savePath = "icon.png";  // Path to save the icon

        SHFILEINFO shfi = new SHFILEINFO();
        IntPtr hIcon = SHGetFileInfo(filePath, 0, ref shfi, (uint)Marshal.SizeOf(shfi), SHGFI_ICON | SHGFI_LARGEICON | SHGFI_USEFILEATTRIBUTES);

        if (hIcon != IntPtr.Zero)
        {
            Icon icon = Icon.FromHandle(shfi.hIcon);
            using (Bitmap bmp = icon.ToBitmap())
            {
                bmp.Save(savePath, System.Drawing.Imaging.ImageFormat.Png);  // Save as PNG
                Console.WriteLine($"Icon saved to {savePath}");
            }
            // Free the icon handle using DestroyIcon
            DestroyIcon(shfi.hIcon);
        }
        else
        {
            Console.WriteLine("Failed to retrieve the icon.");
        }
    }

    [DllImport("user32.dll", SetLastError = true)]
    static extern bool DestroyIcon(IntPtr hIcon);
}
```

### Key Details:

- **Flags in SHGetFileInfo**: The `SHGFI_USEFILEATTRIBUTES` flag allows the function to retrieve information about a file type based on its extension (if a file doesn't exist or is a virtual file, like executables for different file types). This flag helps when you want to retrieve icons for file types rather than specific files.

- **Memory Management**: It's crucial to release the icon handle using `DestroyIcon` after you're done with it to prevent resource leaks.

This solution fetches the icon at the standard large size determined by the system settings, which can be up to 256x256 pixels, but this is dependent on the specifics of the system and how the icons are configured. If you need a different size or more control over the dimensions, you would need to involve additional API functions or manage system image lists more explicitly.
