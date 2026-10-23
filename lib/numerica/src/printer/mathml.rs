//! Safe MathML tokens for coefficient domains without a structural renderer.
use super::{ColorMode, PrintMode, PrintOptions, PrintState};
use crate::domains::Ring;

pub fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

pub fn token(tag: &str, text: &str) -> String {
    format!("<{tag}>{}</{tag}>", escape(text))
}

/// Format an unsigned decimal number as MathML, using a multiplication sign and
/// a power of ten for scientific notation. Preserve the mantissa verbatim;
/// signs on the whole number belong to the enclosing expression.
///
/// Return `None` for text outside decimal/scientific notation.
pub fn number(text: &str) -> Option<String> {
    let decimal = |s: &str| {
        let mut digits = 0;
        let mut dots = 0;
        for c in s.chars() {
            match c {
                '0'..='9' => digits += 1,
                '.' => dots += 1,
                _ => return false,
            }
        }
        digits > 0 && dots <= 1
    };
    if let Some((mantissa, exponent)) = text.split_once(['e', 'E']) {
        let (negative, exponent) = if let Some(rest) = exponent.strip_prefix('-') {
            (true, rest)
        } else {
            (false, exponent.strip_prefix('+').unwrap_or(exponent))
        };
        if !decimal(mantissa)
            || exponent.is_empty()
            || !exponent.bytes().all(|c| c.is_ascii_digit())
        {
            return None;
        }
        // Normalize just the exponent's spelling without parsing into a bounded
        // integer. Even an extreme exponent can be rendered without overflow.
        let exponent = exponent.trim_start_matches('0');
        let exponent = if exponent.is_empty() { "0" } else { exponent };
        let power = if negative && exponent != "0" {
            format!("<mrow><mo>−</mo>{}</mrow>", token("mn", exponent))
        } else {
            token("mn", exponent)
        };
        Some(format!(
            "<mrow>{}<mo>×</mo><msup><mn>10</mn>{power}</msup></mrow>",
            token("mn", mantissa)
        ))
    } else {
        decimal(text).then(|| token("mn", text))
    }
}

/// Preserve exact domain notation as text, while giving ordinary numbers,
/// signs and rational coefficients their mathematical structure.
pub fn coefficient<R: Ring, W: std::fmt::Write>(
    ring: &R,
    element: &R::Element,
    opts: &PrintOptions,
    state: PrintState,
    f: &mut W,
) -> Result<bool, std::fmt::Error> {
    let mut plain = opts.clone();
    plain.mode = PrintMode::Symbolica;
    plain.color_mode = ColorMode::Never;
    plain.num_exp_as_superscript = false;
    plain.max_line_length = None;
    struct BoundedText(String);
    impl std::fmt::Write for BoundedText {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            if self.0.len().saturating_add(value.len()) > 4096 {
                return Err(std::fmt::Error);
            }
            self.0.push_str(value);
            Ok(())
        }
    }
    let mut text = BoundedText(String::new());
    let Ok(suppressed) = ring.format(element, &plain, state, &mut text) else {
        if state.in_sum {
            f.write_str("<mo form=\"infix\">+</mo>")?;
        }
        f.write_str("<mtext>⋯</mtext>")?;
        return Ok(false);
    };
    let text = text.0;
    let (sign, body) = if let Some(rest) = text.strip_prefix('-') {
        (
            if state.in_sum {
                "<mo form=\"infix\">−</mo>"
            } else {
                "<mo>−</mo>"
            },
            rest,
        )
    } else if let Some(rest) = text.strip_prefix('+') {
        ("<mo form=\"infix\">+</mo>", rest)
    } else {
        ("", text.as_str())
    };
    f.write_str(sign)?;
    if !body.is_empty() {
        let integer = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
        if let Some(number) = number(body) {
            f.write_str(&number)?;
        } else if let Some((num, den)) = body.split_once('/')
            && integer(num)
            && integer(den)
        {
            write!(f, "<mfrac>{}{}</mfrac>", token("mn", num), token("mn", den))?;
        } else {
            f.write_str(&token("mtext", body))?;
        }
    }
    Ok(suppressed)
}

#[cfg(test)]
mod tests {
    use super::number;

    #[test]
    fn scientific_notation_preserves_precision_and_structures_exponents() {
        assert_eq!(
            number("4.21359549995700e+1").unwrap(),
            "<mrow><mn>4.21359549995700</mn><mo>×</mo><msup><mn>10</mn><mn>1</mn></msup></mrow>"
        );
        assert_eq!(
            number("1.2300E-003").unwrap(),
            "<mrow><mn>1.2300</mn><mo>×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>3</mn></mrow></msup></mrow>"
        );
        for text in ["1e0", "1e+000", "1E-00"] {
            assert!(number(text).unwrap().contains("<mn>0</mn></msup>"));
        }
        let exponent = "9".repeat(100);
        assert!(
            number(&format!("2e{exponent}"))
                .unwrap()
                .contains(&exponent)
        );
    }

    #[test]
    fn ordinary_decimals_stay_numeric_and_other_text_is_rejected() {
        for text in ["0", "42", "1.2300", ".5", "5."] {
            assert_eq!(number(text).unwrap(), format!("<mn>{text}</mn>"));
        }
        for text in [
            "", ".", "1..2", "1e", "e3", "1e+-2", "1e2e3", "1e2.5", "1E<x>", "-2", "+2", "NaN",
            "inf",
        ] {
            assert!(number(text).is_none(), "{text}");
        }
    }

    #[test]
    fn floating_ring_coefficients_use_the_same_scientific_structure() {
        use crate::domains::{
            Ring,
            float::{Float, FloatField},
        };
        use crate::printer::{PrintOptions, PrintState};

        let field = FloatField::<Float>::new(53);
        let value = Float::parse("-4.21359549995700e+100", Some(53)).unwrap();
        let mut state = PrintState::new();
        state.in_sum = true;
        let mut html = String::new();
        field
            .format_mathml(&value, &PrintOptions::new(), state, &mut html)
            .unwrap();
        assert!(
            html.starts_with("<mo form=\"infix\">−</mo><mrow><mn>4.21359549995700</mn>"),
            "{html}"
        );
        assert!(
            html.contains("<mo>×</mo><msup><mn>10</mn><mn>100</mn>"),
            "{html}"
        );
    }
}
