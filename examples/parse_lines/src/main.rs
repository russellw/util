use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::env;

/// Parse lines from a reader until finding a line that starts with 'q' or reach EOF
/// Returns the total number of lines processed (including the terminating 'q' line if found)
fn parse_lines<R: BufRead>(reader: R) -> io::Result<usize> {
    let mut line_count = 0;

    for line in reader.lines() {
        let line = line?;
        line_count += 1;
        
        // Check if the line starts with 'q'
        if line.starts_with('q') {
            break;
        }
    }

    Ok(line_count)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let line_count: usize;

    if args.len() > 1 {
        // Mode 1: Process lines from a file
        let filename = &args[1];
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        line_count = parse_lines(reader)?;
    } else {
        // Mode 2: REPL mode - process lines from stdin
        let stdin = io::stdin();
        let reader = BufReader::new(stdin.lock());
        line_count = parse_lines(reader)?;
    }

    println!("Total lines processed: {}", line_count);
    Ok(())
}