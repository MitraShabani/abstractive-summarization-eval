import re

HEADING_PATTERN = re.compile(r"^\s*(\d+(\.\d+)*)?\s*[A-Z][A-Za-z\s\-]{2,80}$")

def detect_heading(text, block_font_size, body_text_size, y0, y1, previousBlock_y1, nextBlock_y0, is_bold, is_formula):

    gap_before = (y0 - previousBlock_y1) if previousBlock_y1 is not None else 999
    gap_after = (nextBlock_y0 - y1) if nextBlock_y0 is not None else 999

    def clean_header(t):
        # Digit followed by letter
        t = re.sub(r"(\d)([A-Za-z])", r"\1 \2", t)
        return t.strip()

    text_clean = clean_header(text)

    # exclusion filters
    if is_formula:
        return None
    if block_font_size < body_text_size - 1:  # footnote/caption size
        return None
    if re.match(r"^\s*(Figure|Fig\.|Table)\b", text_clean, re.IGNORECASE):
        return None
    if re.search(r"^\s*(Algorithm|Input:|Output:|Require:|Ensure:)", text_clean, re.IGNORECASE):
        return None
    if re.match(r"^\s*\d+:", text_clean):  # algorithm line numbers like "2: if..."
        return None
    if len(text_clean.strip()) > 300:  # too long to be a header
        return None

    if is_bold:
        return text_clean.strip()

    # Starts with a number or capital letter
    numbered_pattern = re.match(r"^\d+(\.\d+)*[\s.]+\S", text_clean)

    # ALL CAPS
    is_upper = len(text_clean) > 0 and sum(1 for w in text_clean if w[0].isupper()) / len(text_clean) > 0.6  # 3/4 capitalized → True

    if numbered_pattern or is_upper:
        return text_clean


    # Font size check
    big_enough = (block_font_size >= body_text_size * 1.05)  # 5% bigger than body text
    # No period at end
    no_period = not text_clean.endswith(".")
    # Spacing pattern check
    gap_before = (y0 - previousBlock_y1) if previousBlock_y1 is not None else 0
    gap_after = (nextBlock_y0 - y1) if nextBlock_y0 is not None else 999
    spaced = False
    if gap_before and gap_after:
        spaced = ( gap_before > 5) or ( gap_after > 5)


    # strict header
    if big_enough and no_period and spaced:
        return text_clean

    return None




