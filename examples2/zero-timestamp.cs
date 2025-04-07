        // reproducible build
        using (FileStream fs = File.OpenWrite("a.exe"))
        {
            var bytes = new byte[16];

            // constant timestamp
            fs.Seek(0x88, SeekOrigin.Begin);
            fs.Write(bytes, 0, 4);

            // constant MVID
            fs.Seek(0x40c, SeekOrigin.Begin);
            bytes[0] = 1;
            fs.Write(bytes, 0, 16);
        }
