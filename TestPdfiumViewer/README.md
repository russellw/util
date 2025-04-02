# TestPdfiumViewer
Test PdfiumViewer

```
TestPdfiumViewer

Installing:

PdfiumViewer.2.13.0


TestPdfiumViewer

Installing:

Microsoft.Win32.SystemEvents.8.0.0
System.Drawing.Common.8.0.4
```

The errors you're encountering suggest further issues with the approach to use `PdfiumViewer`. The error related to `System.Windows.Forms` indicates that `PdfiumViewer` might require Windows Forms components, which complicates its use in console applications, especially in .NET Core or .NET 5/6+ environments.

Given the complications and potential mismatches in library functionality and .NET versions, I suggest switching to a different approach that is fully compatible with .NET Core and newer .NET versions. A viable alternative is using a combination of libraries that are well-supported and updated regularly.

### Alternative Solution: Using `PdfToImage` Library

One such alternative for rendering PDFs directly to images in .NET Core and newer versions is the `DinkToPdf` library with a wrapper for converting PDF pages to images. Another modern option would be using the `PdfToImage` NuGet package if available, or using `LibVLCSharp` to leverage VLC's capabilities to render PDFs to images, though it's less conventional.

Let's consider a simplified approach using `DinkToPdf`, which is based on the `wkhtmltopdf` tooling and suitable for server-side conversions:

#### Step 1: Install DinkToPdf

1. Add `DinkToPdf` and `DinkToPdf.Contracts` to your project:
   ```bash
   Install-Package DinkToPdf
   Install-Package DinkToPdf.Contracts
   ```

#### Step 2: Configure and Use DinkToPdf

Here's a basic example of using `DinkToPdf` to convert the first page of a PDF into an image:

```csharp
using DinkToPdf;
using DinkToPdf.Contracts;
using System;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        var converter = new BasicConverter(new PdfTools());
        var doc = new HtmlToPdfDocument()
        {
            GlobalSettings = {
                ColorMode = ColorMode.Color,
                Orientation = Orientation.Portrait,
                PaperSize = PaperKind.A4,
            },
            Objects = {
                new ObjectSettings() {
                    PagesCount = true,
                    HtmlContent = @"<html><body><p>This is a test to convert HTML to PDF</p></body></html>",
                    WebSettings = { DefaultEncoding = "utf-8" },
                    HeaderSettings = { FontName = "Arial", FontSize = 9, Right = "Page [page] of [toPage]", Line = true },
                    FooterSettings = { FontName = "Arial", FontSize = 9, Line = true, Center = "Report Footer" }
                }
            }
        };

        byte[] pdf = converter.Convert(doc);

        File.WriteAllBytes("output.pdf", pdf);
        Console.WriteLine("PDF generated successfully.");
    }
}
```

**Note**: This is a basic conversion example using HTML to PDF. To convert PDF to image, you would typically use the output PDF and process it further or use a library like `Magick.NET` to convert PDF pages to images, assuming the PDF content allows direct image extraction.

#### Considerations and Recommendations

- Ensure your project targets a compatible .NET version.
- Given the persistent issues with `PdfiumViewer`, it may be more productive to use a more compatible and straightforward library or tool.
- For PDF to Image conversion in .NET Core, you might also explore external tools like `Ghostscript` or services that can handle this functionality if `DinkToPdf` does not meet your needs directly.

This approach sidesteps the problematic `PdfiumViewer` issues by using a more standardized and widely supported method in the .NET ecosystem, assuming the primary challenge is PDF to image rendering.
