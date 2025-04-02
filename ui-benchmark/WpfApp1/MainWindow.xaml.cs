using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace TypingAssist
{
    public partial class MainWindow : Window
    {
        private Stopwatch stopwatch;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {
            textBox.Clear();
            stopwatch = Stopwatch.StartNew();

            for (int i = 0; i < 10; i++)
            {
                foreach (char letter in "abcdefghijklmnopqrstuvwxyz")
                {
                    textBox.AppendText(letter + " ");
                    textBox.Focus();
                    textBox.CaretIndex = textBox.Text.Length;
                    Application.Current.Dispatcher.Invoke(System.Windows.Threading.DispatcherPriority.Background, new Action(delegate { }));
                    ShowSuggestionBox();
                }
            }

            stopwatch.Stop();
            resultLabel.Content = $"Total time taken: {stopwatch.Elapsed.TotalSeconds:F4} seconds";
        }

        private void TextBox_KeyUp(object sender, KeyEventArgs e)
        {
            ShowSuggestionBox();
        }

        private void ShowSuggestionBox()
        {
            if (textBox.IsFocused)
            {
                int textLength = textBox.Text.Length;
                Rect rect = textBox.GetRectFromCharacterIndex(textLength, true);

                suggestionBox.Margin = new Thickness(rect.Right + 20, rect.Top + 10, 0, 0);
                suggestionBox.Visibility = Visibility.Visible;
            }
        }
    }
}
