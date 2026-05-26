import re

# --- 1. FORMULA DETECTION HELPER ---
def is_formula_block(text, block_font_size, body_text_size):

    text = text.strip()
    if not text:
        return False

    # 1: size (too big to be a formula block)
    if not (body_text_size - 1.0 <= block_font_size <= body_text_size + 0.5):
        return False

    # 2: content (math symbols)
    math_symbols = r"[\=\+\-\*\/\^_{}\[\]\(\)><≈±\u0370-\u03FF\u2200-\u22FF]"
    if not re.search(math_symbols, text):
        return False

    # 3: structure (Centered and short, or contains an equation number)
    is_short = len(text.split()) < 20
    is_equation_number = re.search(r"\s+\(\d{1,3}[a-z]?\s*\)$", text) # e.g., (1), (3a)

    if is_short or is_equation_number:
        # if more symbols than words)
        if len(text) > 5 and (sum(c.isalpha() for c in text) / len(text)) < 0.5:
            return True

    return False