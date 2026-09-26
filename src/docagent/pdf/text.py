"""Text helpers shared by extraction and reconstruction."""

import re

# Only digits, whitespace, punctuation and common units/symbols.
_NUMERIC_RE = re.compile(r"[\d\s.,;:%°+\-–—−()\[\]/×*≈<>~$€£]+")
# Temperature units, frequent in climate reports: "+1,52 °C" is still a number.
_TEMPERATURE_RE = re.compile(r"[°º]\s?[CFK]\b")  # º (U+00BA) is common in IPCC PDFs
# Bullet-like glyphs. Many PDFs encode bullets as control characters (U+0007 in
# the IPCC and HCC files) or private-use code points from symbol fonts.
_BULLETS = set("•◦▪▫■□●○►▶▸‣⁃→✓✔➢➤")
# A line ending with a letter followed by a hyphen: "éco-", "under-" (not "1850-").
_HYPHEN_END_RE = re.compile(r"[^\W\d_]-$")


def join_lines(lines: list[str]) -> str:
    """Join visual lines into one string, undoing end-of-line hyphenation.

    "éco-" + "systèmes" -> "écosystèmes", but "1850-" + "1900" keeps its hyphen
    and a hyphen before an uppercase word ("Nord-" + "Est") is kept too.
    """
    text = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not text:
            text = line
        elif _HYPHEN_END_RE.search(text) and line[0].islower():
            text = text[:-1] + line  # "éco-" + "systèmes"
        elif text.endswith("-"):
            text += line  # "1850-" + "1900", "Nord-" + "Est": keep the hyphen
        else:
            text = f"{text} {line}"
    return re.sub(r"\s+", " ", text)


def is_numeric(text: str) -> bool:
    """True for table values, axis ticks, page numbers: nothing to translate."""
    if not text:
        return False
    stripped = _TEMPERATURE_RE.sub(" ", text)
    return _NUMERIC_RE.fullmatch(stripped) is not None


def is_symbol(char: str) -> bool:
    """Bullets, control characters and private-use glyphs: never translated."""
    code = ord(char)
    return char in _BULLETS or (code < 32 and char not in "\t\n") or 0xE000 <= code <= 0xF8FF


_WORD_RE = re.compile(r"[^\W\d_]{2,}")


def has_words(text: str) -> bool:
    """True if the text has at least one word of 2+ letters.

    Filters out math variables and subscripts ("b", "t", "x_i") that LaTeX
    papers scatter as separate blocks: there is nothing to translate there.
    """
    return _WORD_RE.search(text) is not None


# TeX math fonts (Computer Modern math italic/symbols/extensions, AMS symbols,
# MathType "mwa/mwb" fonts): text set in them is formula, not prose.
_MATH_FONT_RE = re.compile(
    r"^(cmmi|cmsy|cmex|cmbsy|msam|msbm|eufm|rsfs|mw[ab]_|stix.*math)|math", re.I
)


def is_math_font(font: str) -> bool:
    return _MATH_FONT_RE.search(font.split("+", 1)[-1]) is not None


# Inline style markup: "<s1>Au cours de la dernière décennie,</s1> le réchauffement".
# Tags reference entries of TextBlock.styles; the translator must keep them.
TAG_RE = re.compile(r"</?s\d+>")
# HTML tags an LLM may add on its own: "XXI<sup>e</sup> siècle", "CO<sub>2</sub>".
# They are rendered (sub/superscript, bold, italic) instead of being shown as text.
HTML_TAG_RE = re.compile(r"</?(?:sub|sup|b|i|em|strong)>|<br\s*/?>", re.I)


def strip_tags(markup: str) -> str:
    """Plain text of a markup string."""
    return HTML_TAG_RE.sub("", TAG_RE.sub("", markup))


# Style words in font names. Many PDFs don't set the bold/italic flags and only
# say it in the name: "FrutigerLTPro-BlackCn", "FrutigerLTPro-CondensedI".
_BOLD_NAME_RE = re.compile(r"bold|black|heavy|demi|ultra|medi(?!um)", re.I)
# "Italic", "Oblique", or a trailing "I"/"It" after the weight ("CondensedI", "LightCnIt").
_ITALIC_NAME_RE = re.compile(r"(?i:italic|oblique)|(?<=[a-z])(?:I|It)(?:MT)?$")


def font_style(font: str, flags: int) -> tuple[bool, bool]:
    """(bold, italic) from a span's flags and its font name."""
    style = font.split("+", 1)[-1].split("-", 1)[-1] if "-" in font else ""
    bold = bool(flags & 16) or bool(_BOLD_NAME_RE.search(style))
    italic = bool(flags & 2) or bool(_ITALIC_NAME_RE.search(style))
    return bold, italic
