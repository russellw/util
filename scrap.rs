
fn report_char(c: char) -> String {
    if c.is_ascii_graphic() {
        c.to_string()
    } else {
        format!("\\u{:x}", c as u32)
    }
}

    fn peek_eq(&self) -> bool {
        let mut i = self.pos;
        while self.chars[i] == ' ' || self.chars[i] == '\t' {
            i += 1;
        }
        if self.chars[i] != '=' {
            return false;
        }
        match self.chars[i + 1] {
            '=' | '>' | '<' => false,
            _ => true,
        }
    }
