using Microsoft.VisualBasic.FileIO;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace CsvDelCols
{
    class Program
    {
        static IEnumerable<string[]> ReadCsv(string file)
        {
            // For some obscure historical reason, the CSV parser is in the Visual Basic library
            // So to make this work in a new project, right click project, Add Reference, Microsoft.VisualBasic
            using (TextFieldParser parser = new TextFieldParser(file))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");
                while (!parser.EndOfData)
                    yield return parser.ReadFields();
            }
        }

        static void Main(string[] args)
        {
            string file = null;
            var cols = new HashSet<int>();
            for (int i = 0; i < args.Length; i++)
            {
                var s = args[i];

                // On Windows, options can start with /
                if (Path.DirectorySeparatorChar == '\\' && s.StartsWith("/"))
                    s = "-" + s.Substring(1);

                // Not an option
                if (!s.StartsWith("-"))
                {
                    if (file == null)
                    {
                        file = s;
                        continue;
                    }
                    int c = 0;
                    if (char.IsDigit(s[0]))
                    {
                        try
                        {
                            c = int.Parse(s);
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine(args[i] + ": " + e.Message);
                            Environment.Exit(1);
                        }
                    }
                    else if (char.IsLetter(s[0]))
                    {
                        if (s.Length > 1)
                        {
                            Console.WriteLine(args[i] + ": Use numbers past column Z");
                            Environment.Exit(1);
                        }
                        c = char.ToLower(s[0]) - 'a';
                    }
                    else
                    {
                        Console.WriteLine(args[i] + ": Expected column");
                        Environment.Exit(1);
                    }
                    cols.Add(c);
                    continue;
                }

                // Option
                s = s.TrimStart('-');
                switch (s)
                {
                    case "?":
                    case "h":
                    case "help":
                        Console.WriteLine("Options:");
                        Console.WriteLine();
                        Console.WriteLine("-h  Show help");
                        Console.WriteLine("-v  Show version");
                        return;
                    case "V":
                    case "v":
                    case "version":
                        var assemblyName = Assembly.GetExecutingAssembly().GetName();
                        Console.WriteLine("{0} {1}", assemblyName.Name, assemblyName.Version.ToString(2));
                        return;
                    default:
                        Console.WriteLine(args[i] + ": Unknown option");
                        Environment.Exit(1);
                        break;
                }
            }
            if (file == null)
            {
                var assemblyName = Assembly.GetExecutingAssembly().GetName();
                Console.WriteLine(string.Format("Usage: {0} <file> <columns>", assemblyName.Name));
                Environment.Exit(1);
            }
            try
            {
                foreach (var line in ReadCsv(file))
                {
                    Console.WriteLine(string.Join(",", 
                        Enumerable.Range(0, line.Length).Where(i => !cols.Contains(i)).Select(i => line[i])));
                }
            }
            catch (IOException e)
            {
                Console.WriteLine(e.Message);
                Environment.Exit(1);
            }
        }
    }
}
