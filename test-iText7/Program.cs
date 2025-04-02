using System;
using System.IO;
using Ghostscript.NET.Rasterizer;
using iText.Kernel.Pdf;
using System.Drawing.Imaging; // Make sure this using directive is added

class Program
{
    static void Main(string[] args)
    {
        string pdfPath = @"path_to_your_pdf.pdf";
        string outputDirectory = @"thumbnails";
        Directory.CreateDirectory(outputDirectory);

        using (var rasterizer = new GhostscriptRasterizer())
        {
            rasterizer.Open(pdfPath);

            for (int pageNumber = 1; pageNumber <= rasterizer.PageCount; pageNumber++)
            {
                using (var img = rasterizer.GetPage(96, 96, pageNumber)) // Use DPI settings for X and Y
                {
                    string thumbnailPath = Path.Combine(outputDirectory, $"thumbnail_{pageNumber}.png");
                    img.Save(thumbnailPath, ImageFormat.Png);
                    Console.WriteLine($"Thumbnail created at: {thumbnailPath}");
                }
            }
        }
    }
}
