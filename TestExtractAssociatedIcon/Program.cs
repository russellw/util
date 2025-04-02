using System;
using System.Drawing;

class Program
{
    static void Main(string[]args)
    {
        string filePath = args[0];
        string savePath = @"icon.png"; // Path to save the icon

        try
        {
            Icon icon = Icon.ExtractAssociatedIcon(filePath);
            if (icon != null)
            {
                using (Bitmap bmp = icon.ToBitmap())
                {
                    bmp.Save(savePath, System.Drawing.Imaging.ImageFormat.Png); // Save as PNG
                }
                Console.WriteLine($"Icon saved to {savePath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("An error occurred: " + ex.Message);
        }
    }
}
