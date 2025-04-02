using UglyToad.PdfPig;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Formats.Jpeg;
using System;
using System.IO;
using System.Linq;

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
        using (var document = PdfDocument.Open(pdfPath))
        {
            var page = document.GetPage(1); // Get the first page
            var images = page.GetImages().ToList(); // Get all images from the page

            if (images.Any())
            {
                using (var image = Image.Load(images.First().Bytes))
                {
                    image.Mutate(x => x.Resize(120, 120)); // Resize to thumbnail size
                    image.Save(outputPath, new JpegEncoder()); // Save as JPEG
                }
            }
            else
            {
                Console.WriteLine("No images found on the first page to generate thumbnail.");
            }
        }
    }
}
