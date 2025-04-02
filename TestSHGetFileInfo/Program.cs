using System;
using System.Drawing;
using Microsoft.WindowsAPICodePack.Shell;

class Program
{
    public static Bitmap GetLargeThumbnail(string path)
    {
        ShellObject shellObject = ShellObject.FromParsingName(path);
        // This will retrieve the largest available thumbnail for the file
        return shellObject.Thumbnail.ExtraLargeBitmap;
    }

    static void Main(string[]args)
    {
        string filePath = args[0];
        string savePath = @"icon.png";

        Bitmap largeThumbnail = GetLargeThumbnail(filePath);
        if (largeThumbnail != null)
        {
            largeThumbnail.Save(savePath, System.Drawing.Imaging.ImageFormat.Png);
            Console.WriteLine($"Thumbnail saved to {savePath}");
        }
        else
        {
            Console.WriteLine("Failed to retrieve the large thumbnail.");
        }
    }
}
