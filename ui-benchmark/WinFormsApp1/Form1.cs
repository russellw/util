using System;
using System.Diagnostics;
using System.Drawing;
using System.Windows.Forms;

namespace TypingAssist
{
    public partial class Form1 : Form
    {
        private TextBox textBox;
        private Label resultLabel;
        private Stopwatch stopwatch;
        private ListBox suggestionBox;

        public Form1()
        {
            InitializeCustomComponents();
        }

        private void InitializeCustomComponents()
        {
            // Set up the form
            this.Text = "Typing Assist Program Benchmark";
            this.Size = new Size(800, 600);

            // Create and configure the TextBox
            this.textBox = new TextBox
            {
                Multiline = true,
                Width = 750,
                Height = 400,
                Location = new Point(10, 10)
            };
            this.textBox.KeyUp += TextBox_KeyUp;
            this.Controls.Add(this.textBox);

            // Create and configure the start button
            Button startButton = new Button
            {
                Text = "Start Insertion",
                Location = new Point(10, 420)
            };
            startButton.Click += StartButton_Click;
            this.Controls.Add(startButton);

            // Create and configure the result label
            this.resultLabel = new Label
            {
                Location = new Point(120, 425),
                AutoSize = true
            };
            this.Controls.Add(this.resultLabel);

            // Create and configure the suggestion box
            this.suggestionBox = new ListBox
            {
                Visible = false,
                Width = 100,
                Height = 80
            };
            this.suggestionBox.Items.AddRange(new object[] {
                "1. Lenovo",
                "2. HP",
                "3. Dell",
                "4. Apple",
                "5. Asus"
            });
            this.Controls.Add(this.suggestionBox);
        }

        private void TextBox_KeyUp(object sender, KeyEventArgs e)
        {
            ShowSuggestionBox();
        }

        private void StartButton_Click(object sender, EventArgs e)
        {
            this.textBox.Clear();
            this.stopwatch = Stopwatch.StartNew();

            for (int i = 0; i < 10; i++)
            {
                foreach (char letter in "abcdefghijklmnopqrstuvwxyz")
                {
                    this.textBox.AppendText(letter + " ");
                    Application.DoEvents(); // Allow the UI to update
                    ShowSuggestionBox();
                }
            }

            this.stopwatch.Stop();
            this.resultLabel.Text = $"Total time taken: {this.stopwatch.Elapsed.TotalSeconds:F4} seconds";
        }

        private void ShowSuggestionBox()
        {
            if (this.textBox.Focused)
            {
                //TODO: figure out why this still doesn't really work
                // Calculate the position of the cursor based on the length of the text
                int textLength = this.textBox.TextLength;
                Point cursorPosition = this.textBox.GetPositionFromCharIndex(textLength);
                Point textBoxPosition = this.textBox.PointToScreen(cursorPosition);

                // Adjust position for the suggestion box
                this.suggestionBox.Location = new Point(textBoxPosition.X + 20, textBoxPosition.Y);
                this.suggestionBox.Visible = true;
                this.suggestionBox.BringToFront();
            }
        }
    }
}
