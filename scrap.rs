
fn report_char(c: char) -> String {
    if c.is_ascii_graphic() {
        c.to_string()
    } else {
        format!("\\u{:x}", c as u32)
    }
}
