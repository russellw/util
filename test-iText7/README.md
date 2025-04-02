# test-iText7
Test iText7

```
test-iText7

Installing:

itext.8.0.4
itext.commons.8.0.4
itext7.8.0.4
Microsoft.DotNet.PlatformAbstractions.1.1.0
Microsoft.Extensions.DependencyInjection.5.0.0
Microsoft.Extensions.DependencyInjection.Abstractions.5.0.0
Microsoft.Extensions.DependencyModel.1.1.0
Microsoft.Extensions.Logging.5.0.0
Microsoft.Extensions.Logging.Abstractions.5.0.0
Microsoft.Extensions.Options.5.0.0
Microsoft.Extensions.Primitives.5.0.0
Microsoft.NETCore.Platforms.1.1.1
Microsoft.NETCore.Targets.1.1.3
Microsoft.Win32.Primitives.4.3.0
Microsoft.Win32.Registry.4.3.0
Newtonsoft.Json.13.0.1
runtime.debian.8-x64.runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
runtime.fedora.23-x64.runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
runtime.fedora.24-x64.runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
runtime.native.System.4.3.0
runtime.native.System.Security.Cryptography.Apple.4.3.0
runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
runtime.opensuse.13.2-x64.runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
runtime.opensuse.42.1-x64.runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
runtime.osx.10.10-x64.runtime.native.System.Security.Cryptography.Apple.4.3.0
runtime.osx.10.10-x64.runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
runtime.rhel.7-x64.runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
runtime.ubuntu.14.04-x64.runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
runtime.ubuntu.16.04-x64.runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
runtime.ubuntu.16.10-x64.runtime.native.System.Security.Cryptography.OpenSsl.4.3.0
System.AppContext.4.1.0
System.Collections.4.3.0
System.Collections.Concurrent.4.3.0
System.Collections.NonGeneric.4.3.0
System.Diagnostics.Debug.4.3.0
System.Diagnostics.Process.4.3.0
System.Diagnostics.Tracing.4.3.0
System.Dynamic.Runtime.4.0.11
System.Globalization.4.3.0
System.Globalization.Extensions.4.3.0
System.IO.4.3.0
System.IO.FileSystem.4.3.0
System.IO.FileSystem.Primitives.4.3.0
System.Linq.4.3.0
System.Linq.Expressions.4.1.0
System.ObjectModel.4.0.12
System.Reflection.4.3.0
System.Reflection.Emit.4.0.1
System.Reflection.Emit.ILGeneration.4.0.1
System.Reflection.Emit.Lightweight.4.0.1
System.Reflection.Extensions.4.0.1
System.Reflection.Primitives.4.3.0
System.Reflection.TypeExtensions.4.1.0
System.Resources.ResourceManager.4.3.0
System.Runtime.4.3.1
System.Runtime.Extensions.4.3.0
System.Runtime.Handles.4.3.0
System.Runtime.InteropServices.4.3.0
System.Runtime.InteropServices.RuntimeInformation.4.0.0
System.Runtime.Loader.4.3.0
System.Runtime.Numerics.4.3.0
System.Runtime.Serialization.Formatters.4.3.0
System.Runtime.Serialization.Primitives.4.3.0
System.Security.Cryptography.Algorithms.4.3.0
System.Security.Cryptography.Csp.4.3.0
System.Security.Cryptography.Encoding.4.3.0
System.Security.Cryptography.Primitives.4.3.0
System.Text.Encoding.4.3.0
System.Text.Encoding.CodePages.4.3.0
System.Text.Encoding.Extensions.4.3.0
System.Text.RegularExpressions.4.3.1
System.Threading.4.3.0
System.Threading.Tasks.4.3.0
System.Threading.Tasks.Extensions.4.3.0
System.Threading.Thread.4.3.0
System.Threading.ThreadPool.4.3.0
System.Xml.ReaderWriter.4.3.0
System.Xml.XmlDocument.4.3.0


test-iText7

Installing:

Ghostscript.NET.1.2.3.1


test-iText7

Installing:

Microsoft.Win32.SystemEvents.8.0.0
System.Drawing.Common.8.0.4
```

I apologize for the confusion in the previous code snippet. It appears that there was a mistake in the method signature for `GetPage` from the `GhostscriptRasterizer` class. Let's correct this and ensure the code accurately renders PDF pages to images using Ghostscript.NET.

### Correct Usage of `GetPage`

The correct usage of the `GetPage` method requires different parameters. Here is how you can correctly utilize the `GhostscriptRasterizer` to render thumbnails:

```csharp
using System;
using System.IO;
using Ghostscript.NET.Rasterizer;
using System.Drawing;
using System.Drawing.Imaging;

class Program
{
    static void Main(string[] args)
    {
        string pdfPath = @"path_to_your_pdf.pdf";
        string outputDirectory = @"thumbnails";
        Directory.CreateDirectory(outputDirectory);

        using (var rasterizer = new GhostscriptRasterizer())
        {
            rasterizer.Open(pdfPath, Ghostscript.NET.GhostscriptVersionInfo.GetLastInstalledVersion(), false);

            for (int pageNumber = 1; pageNumber <= rasterizer.PageCount; pageNumber++)
            {
                using (Image img = rasterizer.GetPage(96, 96, pageNumber))
                {
                    string thumbnailPath = Path.Combine(outputDirectory, $"thumbnail_{pageNumber}.png");
                    img.Save(thumbnailPath, ImageFormat.Png);
                    Console.WriteLine($"Thumbnail created at: {thumbnailPath}");
                }
            }
        }
    }
}
```

### Key Changes & Notes:

- **Ghostscript Initialization**: The `rasterizer.Open` method is called with the `GhostscriptVersionInfo.GetLastInstalledVersion()` to ensure the correct version of Ghostscript is used.
- **Image Handling**: We explicitly use `System.Drawing.Image` for the `img` object. This is part of the `System.Drawing.Common` package in .NET Core or .NET Framework projects.
- **Resolution Parameters**: The `GetPage` method is now correctly called with parameters for X and Y resolution set to 96 DPI, and the page number.

### Before Running the Program:

1. **Ensure Ghostscript is Installed**: Ghostscript must be installed on your machine, as Ghostscript.NET acts as a wrapper around the Ghostscript application.
2. **Check `System.Drawing.Common`**: Make sure that the `System.Drawing.Common` package is correctly installed in your project.
3. **Verify PDF Path**: Ensure the path to the PDF file and the output directory is correct.

Run your corrected program. This should now function correctly, generating thumbnails from each page of the specified PDF document using Ghostscript.NET. If there are further issues, it may be helpful to check for compatibility issues or additional dependencies related to Ghostscript.NET.
