import re

def fix_hyphenation(text):

    """ cleaning hyphens within a block before spaCy """

    # remove hyphen + newline breaks like:
    #   "self-\nidentify" → "selfidentify"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # also remove hyphen at end of line when block extraction puts the line in one string:
    # "self-" + "identify" → "selfidentify"
    text = re.sub(r"(\w)-\s*(\w)", r"\1\2", text)

    return text

def remove_references(text):
    # Cut everything from "References" section onward
    match = re.search(r'\n\s*References\s*\n', text, re.IGNORECASE)
    if match:
        text = text[:match.start()]
    return text

def clean_text(text):
    text = fix_hyphenation(text)
    text = remove_references(text)
    # Remove arXiv stamp
    text = re.sub(r'arXiv:\S+\s+\[.*?\].*', '', text)
    # Remove license notice
    text = re.sub(r'Licensed under.*?reserved\.', '', text, flags=re.DOTALL)
    # Remove LaTeX markup
    text = re.sub(r'[#@\\][a-zA-Z0-9]+', '', text)
    # Remove very short lines (less than 3 words)
    lines = text.split('\n')
    lines = [l for l in lines if len(l.split()) >= 3]
    return ' '.join(lines)

