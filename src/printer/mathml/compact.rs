//! Bounded width estimates for small semantic MathML fragments. These are glyph
//! estimates, not browser measurements; unknown layout instructions stay folded.

const GLYPH: usize = 10;
const MAX_BYTES: usize = 4096;
const MAX_DEPTH: usize = 64;

pub(super) fn fits(fragment: &str, glyph_budget: usize) -> bool {
    if fragment.len() > MAX_BYTES {
        return false;
    }
    let mut parser = Parser {
        remaining: fragment,
    };
    let mut width = 0;
    while !parser.remaining.trim().is_empty() {
        parser.whitespace();
        let Some(part) = parser.element(0, 0) else {
            return false;
        };
        width += part;
        if width > glyph_budget.saturating_mul(GLYPH) {
            return false;
        }
    }
    true
}

struct Parser<'a> {
    remaining: &'a str,
}

impl<'a> Parser<'a> {
    fn whitespace(&mut self) {
        self.remaining = self
            .remaining
            .trim_start_matches(|c: char| c.is_ascii_whitespace());
    }

    fn consume(&mut self, text: &str) -> Option<()> {
        self.remaining = self.remaining.strip_prefix(text)?;
        Some(())
    }

    fn name(&mut self) -> Option<&'a str> {
        let end = self
            .remaining
            .find(|c: char| !c.is_ascii_alphanumeric() && !matches!(c, '-' | '_' | ':'))
            .unwrap_or(self.remaining.len());
        let (name, remaining) = self.remaining.split_at(end);
        if name.is_empty() {
            return None;
        }
        self.remaining = remaining;
        Some(name)
    }

    fn attribute(&mut self) -> Option<(&'a str, &'a str)> {
        let name = self.name()?;
        self.whitespace();
        self.consume("=")?;
        self.whitespace();
        let quote = self.remaining.chars().next()?;
        if !matches!(quote, '\'' | '"') {
            return None;
        }
        self.remaining = &self.remaining[1..];
        let end = self.remaining.find(quote)?;
        let value = &self.remaining[..end];
        if value.contains(['<', '&']) {
            return None;
        }
        self.remaining = &self.remaining[end + 1..];
        Some((name, value))
    }

    fn element(&mut self, depth: usize, script_depth: usize) -> Option<usize> {
        // Browser minimum script sizes eventually stop shrinking. Keep the
        // estimate to the two script levels used by indexed tensor labels.
        if depth >= MAX_DEPTH || script_depth > 2 {
            return None;
        }
        self.consume("<")?;
        let name = self.name()?;
        let token = matches!(name, "mi" | "mn" | "mo" | "mtext" | "ms");
        if !token
            && !matches!(
                name,
                "math"
                    | "mrow"
                    | "mstyle"
                    | "msub"
                    | "msup"
                    | "msubsup"
                    | "mmultiscripts"
                    | "mfrac"
                    | "none"
                    | "mspace"
            )
        {
            return None;
        }
        let mut space = 0;
        let mut attributes = Vec::new();
        let empty = loop {
            let before = self.remaining.len();
            self.whitespace();
            if self.remaining.starts_with("/>") {
                self.consume("/>")?;
                break true;
            }
            if self.remaining.starts_with('>') {
                self.consume(">")?;
                break false;
            }
            if before == self.remaining.len() {
                return None;
            }
            let (attribute, value) = self.attribute()?;
            if attributes.contains(&attribute) {
                return None;
            }
            attributes.push(attribute);
            match (attribute, value) {
                ("width", _) if name == "mspace" => space = Self::space(value)?,
                ("lspace" | "rspace", "0" | "0em") if name == "mo" => (),
                ("stretchy", "false") if name == "mo" => (),
                ("form", "prefix" | "infix" | "postfix") if name == "mo" => (),
                ("mathvariant", "normal" | "italic" | "bold" | "bold-italic") if token => (),
                ("displaystyle", "true" | "false") => (),
                ("xmlns", "http://www.w3.org/1998/Math/MathML") if name == "math" => (),
                ("data-symbolica-path", _) => (),
                _ => return None,
            }
        };
        let mut children = Vec::new();
        let mut text_width = 0;
        if !empty {
            loop {
                if self.remaining.starts_with("</") {
                    self.consume("</")?;
                    if self.name()? != name {
                        return None;
                    }
                    self.whitespace();
                    self.consume(">")?;
                    break;
                }
                if self.remaining.starts_with('<') {
                    if token {
                        return None;
                    }
                    let script = !children.is_empty()
                        && matches!(name, "msub" | "msup" | "msubsup" | "mmultiscripts");
                    children.push(self.element(depth + 1, script_depth + usize::from(script))?);
                } else {
                    let end = self.remaining.find('<')?;
                    let text = &self.remaining[..end];
                    if token {
                        text_width += Self::text(text)?;
                    } else if !text.trim().is_empty() {
                        return None;
                    }
                    self.remaining = &self.remaining[end..];
                }
            }
        }
        if token {
            return Some(text_width + if name == "ms" { 2 * GLYPH } else { 0 });
        }
        match (name, children.as_slice()) {
            ("none", []) => Some(0),
            ("mspace", []) => Some(space),
            ("math" | "mrow" | "mstyle", _) => Some(children.iter().sum()),
            ("msub" | "msup", [base, script]) => Some(base + Self::script(*script)),
            ("msubsup", [base, sub, sup]) => Some(base + Self::script((*sub).max(*sup))),
            ("mmultiscripts", [base, scripts @ ..]) if scripts.len() % 2 == 0 => Some(
                base + scripts
                    .chunks_exact(2)
                    .map(|pair| Self::script(pair[0].max(pair[1])))
                    .sum::<usize>(),
            ),
            ("mfrac", [numerator, denominator]) => Some(numerator.max(denominator) + 2),
            _ => None,
        }
    }

    fn script(width: usize) -> usize {
        (width * 7).div_ceil(10)
    }

    fn space(value: &str) -> Option<usize> {
        if value == "0" {
            return Some(0);
        }
        let value = value.strip_suffix("em")?;
        if value.is_empty() || value == "." {
            return None;
        }
        let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
        if fraction.len() > 3 || !fraction.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let whole = if whole.is_empty() {
            0
        } else {
            whole.parse::<usize>().ok()?
        };
        if whole > 100 {
            return None;
        }
        let divisor = 10_usize.pow(fraction.len() as u32);
        let fraction = if fraction.is_empty() {
            0
        } else {
            fraction.parse::<usize>().ok()?
        };
        // One em is approximately two ordinary mathematical glyphs.
        Some(whole * 2 * GLYPH + (fraction * 2 * GLYPH).div_ceil(divisor))
    }

    fn text(mut text: &str) -> Option<usize> {
        let mut width = 0;
        while let Some(character) = text.chars().next() {
            text = &text[character.len_utf8()..];
            let character = if character == '&' {
                let end = text.find(';')?;
                let entity = &text[..end];
                text = &text[end + 1..];
                match entity {
                    "amp" => '&',
                    "lt" => '<',
                    "gt" => '>',
                    "quot" => '"',
                    "apos" => '\'',
                    _ => {
                        let code = if let Some(hex) = entity.strip_prefix("#x") {
                            u32::from_str_radix(hex, 16).ok()?
                        } else {
                            entity.strip_prefix('#')?.parse().ok()?
                        };
                        char::from_u32(code)?
                    }
                }
            } else {
                character
            };
            if !matches!(character, '\u{2061}'..='\u{2064}') {
                width += GLYPH;
            }
        }
        Some(width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_metric_fits_without_counting_serialized_metadata() {
        let metric = "<mmultiscripts><mrow><mi>g</mi></mrow><none/>
            <mrow><msub><mi>ρ</mi><mn>2</mn></msub></mrow>
            <mrow><mspace width=\"0.08em\"/></mrow>
            <mrow><mspace width=\"0.08em\"/></mrow><none/>
            <mrow><msub><mi>ν</mi><mn>2</mn></msub></mrow></mmultiscripts>";
        assert!(fits(metric, 4));
        assert!(!fits(metric, 3));
    }

    #[test]
    fn stacked_scripts_and_fractions_use_the_widest_column() {
        let scripts = "<msubsup><mi>x</mi><mn>1234</mn><mn>5678</mn></msubsup>";
        assert!(fits(scripts, 4));
        assert!(!fits(scripts, 3));
        assert!(fits("<mfrac><mn>123</mn><mn>456</mn></mfrac>", 4));
    }

    #[test]
    fn indentation_entities_and_token_spaces_are_distinct() {
        assert!(fits(
            "<mrow>\n <mi>&#x3c1;</mi>\n <mo>+</mo><mi>x</mi>\n</mrow>",
            3
        ));
        assert!(fits("<mi>x</mi><mo>&#x2062;</mo><mi>y</mi>", 2));
        assert!(!fits("<mtext>a b</mtext>", 2));
        assert!(!fits("<mi>long_name</mi>", 4));
        assert!(!fits("<mspace width=\"5em\"/>", 4));
    }

    #[test]
    fn unknown_layouts_sizes_and_malformed_fragments_stay_folded() {
        for fragment in [
            "<mi style=\"font-size:100px\">x</mi>",
            "<mi mathsize=\"100px\">x</mi>",
            "<mstyle scriptlevel=\"-3\"><mi>x</mi></mstyle>",
            "<mtable><mtr><mtd><mi>x</mi></mtd></mtr></mtable>",
            "<mspace width=\"10px\"/>",
            "<mspace width=\"100em\" width=\"0em\"/>",
            "<mi>&unknown;</mi>",
            "<mi>x</mn>",
            "<msub><mi>x</mi></msub>",
            "<msup><mi>x</mi><msup><mi>y</mi><msup><mi>z</mi><mi>w</mi></msup></msup></msup>",
            "<mrow>unwrapped text</mrow>",
        ] {
            assert!(!fits(fragment, 4), "{fragment}");
        }
    }

    #[test]
    fn input_size_and_recursion_are_bounded() {
        assert!(!fits(
            &format!("<mi>{}</mi>", "x".repeat(MAX_BYTES)),
            usize::MAX
        ));
        let nested = format!(
            "{}<mi>x</mi>{}",
            "<mrow>".repeat(MAX_DEPTH),
            "</mrow>".repeat(MAX_DEPTH)
        );
        assert!(!fits(&nested, 4));
    }
}
