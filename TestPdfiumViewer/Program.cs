using PdfiumViewer;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Enter the path to the PDF file:");
        string pdfPath = Console.ReadLine();

        try
        {
            GenerateThumbnail(pdfPath, "thumbnail.jpg");
            Console.WriteLine("Thumbnail generated successfully.");
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error generating thumbnail: " + ex.Message);
        }
    }

    static void GenerateThumbnail(string pdfPath, string outputPath)
    {
        using (var document = PdfDocument.Load(pdfPath))
        {
            // Create a PDF renderer with the document
            var renderer = new PdfRenderer(document);
            // Render the first page at 300 DPI
            using (var image = renderer.Render(0, 300, 300, PdfRenderFlags.CorrectFromDpi))
            {
                // Resize the image to a thumbnail size
                var thumbnail = ResizeImage(image, 120, 120);
                thumbnail.Save(outputPath, ImageFormat.Jpeg);
            }
        }
    }

    static Bitmap ResizeImage(Image image, int width, int height)
    {
        var destRect = new Rectangle(0, 0, width, height);
        var destImage = new Bitmap(width, height);

        destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

        using (var graphics = Graphics.FromImage(destImage))
        {
            graphics.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
            graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;

            using (var wrapMode = new ImageAttributes())
            {
                wrapMode.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY);
                graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
            }
        }

        return destImage;
    }
}
