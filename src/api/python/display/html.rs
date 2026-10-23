//! Adapt root MathML flows for static HTML, without changing MathML exports or
//! page payloads. Token spans preserve custom notation, entities and attributes.
use std::ops::Range;
use xmlparser::{ElementEnd, Token, Tokenizer};

pub(super) fn wrapping_mathml(html: &str) -> Option<String> {
    let mut depth = 0usize;
    let mut name = "";
    let mut start = 0;
    let mut attributes = 0;
    let mut root_attributes = String::new();
    let mut row_attributes = "";
    let mut style = "";
    let mut term = (0, 0, false);
    let mut terms: Vec<(Range<usize>, Range<usize>, bool)> = Vec::new();
    for token in Tokenizer::from(html) {
        match token.ok()? {
            Token::ElementStart {
                prefix,
                local,
                span,
            } => {
                if !prefix.is_empty() {
                    return None;
                }
                name = local.as_str();
                start = span.start();
                attributes = span.end();
            }
            Token::Attribute {
                local, value, span, ..
            } => {
                if depth == 0 && local.as_str() != "xmlns" {
                    root_attributes.push(' ');
                    root_attributes.push_str(span.as_str());
                }
                if depth == 1 && local.as_str() == "style" {
                    style = value.as_str();
                }
            }
            Token::ElementEnd {
                end: ElementEnd::Open,
                span,
            } => {
                match depth {
                    0 if name != "math" => return None,
                    1 => {
                        let properties: Vec<_> = style
                            .split(';')
                            .filter_map(|p| p.split_once(':'))
                            .map(|(k, v)| (k.trim(), v.trim()))
                            .collect();
                        if name != "mrow"
                            || !properties.contains(&("flex-wrap", "wrap"))
                            || !properties
                                .iter()
                                .any(|p| matches!(*p, ("display", "flex" | "inline-flex")))
                        {
                            return None;
                        }
                        row_attributes = &html[attributes..span.start()];
                    }
                    2 => term = (start, span.end(), name == "mrow"),
                    _ => (),
                }
                depth += 1;
            }
            Token::ElementEnd {
                end: ElementEnd::Empty,
                span,
            } if depth == 2 => {
                terms.push((start..span.end(), 0..0, false));
            }
            Token::ElementEnd {
                end: ElementEnd::Close(_, _),
                span,
            } => {
                depth = depth.checked_sub(1)?;
                if depth == 2 {
                    terms.push((term.0..span.end(), term.1..span.start(), term.2));
                }
            }
            Token::Text { text } if depth == 2 && !text.as_str().trim().is_empty() => return None,
            _ => (),
        }
    }
    if depth != 0 || terms.is_empty() {
        return None;
    }
    let mut result = format!(
        "<div class=\"symbolica-math-flow\"{root_attributes}><style>.symbolica-math-flow>div{{max-width:100%!important}}.symbolica-math-flow>div>math{{font:inherit;font-synthesis:none;flex:none}}</style><div{row_attributes}>"
    );
    for (whole, content, row) in terms {
        if row {
            // Replace the term wrapper rather than adding a second one.
            result.push_str("<math");
            result.push_str(&html[whole.start + "<mrow".len()..content.start]);
            result.push_str(&html[content]);
        } else {
            result.push_str("<math>");
            result.push_str(&html[whole]);
        }
        result.push_str("</math>");
    }
    result.push_str("</div></div>");
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_flow_preserves_terms_and_custom_attributes() {
        let input = "<math xmlns=\"http://www.w3.org/1998/Math/MathML\" style=\"font-size:21px\"><mrow data-symbolica-flow=\"\" style=\"display:flex;flex-wrap:wrap\"><mrow title=\"a > b\"><mi>x</mi></mrow><mspace width=\".5em\"/><mrow><mo>+</mo><mn>2</mn></mrow></mrow></math>";
        let html = wrapping_mathml(input).unwrap();
        assert!(html.contains("<math title=\"a > b\"><mi>x</mi></math>"));
        assert!(html.contains("<math><mspace width=\".5em\"/></math>"));
        assert!(html.contains("data-symbolica-flow=\"\""));
        assert_eq!(html.matches("<math").count(), 3);
        assert!(!html.contains("xmlns="));
    }

    #[test]
    fn nested_or_non_flow_math_is_unchanged() {
        for html in [
            "<math><mi>x</mi></math>",
            "<math><msup><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mn>2</mn></msup></math>",
            "<math><mrow style=\"display:flex\"><mi>x</mi></mrow></math>",
            "<math>",
        ] {
            assert!(wrapping_mathml(html).is_none(), "{html}");
        }
    }
}
