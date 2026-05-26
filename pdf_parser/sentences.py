import spacy
import re
import fitz
from .headers import detect_heading
from .formula import is_formula_block

# Load language model once
nlp = spacy.load("en_core_web_sm")


def fix_hyphenation(text):

    """ cleaning hyphens within a block before spaCy """

    # remove hyphen + newline breaks like:
    #   "self-\nidentify" → "selfidentify"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # also remove hyphen at end of line when block extraction puts the line in one string:
    # "self-" + "identify" → "selfidentify"
    text = re.sub(r"(\w)-\s*(\w)", r"\1\2", text)

    return text

def repair_data(page_sentences, page_number):
        # HYPHENATION
        repaired_data = []
        i = 0

        while i < len(page_sentences):
            current = page_sentences[i]

            # hold the combined text if a merge occurs
            merged_text = None

            # skip Formulas/Headers entirely in the 'while' loop
            if current.get("is_formula") or current.get("is_header"):
                repaired_data.append({
                    "page": page_number,
                    "sentence": current["sentence"],
                    "header": current["header"],
                    "type": "formula" if current.get("is_formula") else "header"
                })
                i += 1
                continue

            # to prevent the indexError when the loop reaches the last element in the 'page_sentences' list
            if i + 1 < len(page_sentences):
                next_sent = page_sentences[i+1]

                # Do not merge into the next header or formula!
                if next_sent.get("is_header") or next_sent.get("is_formula"):
                    pass

                # 1: Hyphenation Fix
                elif current["sentence"].endswith('-'):

                    # merge the words by removing the hyphen and joining immediately
                    s1_clean = current["sentence"].rstrip('-').rstrip()
                    s2 = next_sent["sentence"]
                    merged_text = f"{s1_clean}{s2}"
                    # Consolidate header context for the merged block
                    merged_header = current["header"] or next_sent["header"]
                    i += 2   # skip both current and next (because they are merged)

                # 2: Punctuation Fix
                elif (current["sentence"].endswith(('.', '?', '!', ':')) and
                    next_sent["sentence"] and
                    next_sent["sentence"][0].isalpha()
                    and not next_sent["sentence"][0].isupper()):

                    # If a sentence ends with punctuation and the next one starts lowercase, it means spaCy missed something. Join them with a space.
                    merged_text = f"{current['sentence']} {next_sent['sentence']}"
                    merged_header = current["header"] or next_sent["header"]
                    i += 2

            # process Current or Merged Text
            if merged_text:
                # If text was merged, pass the *new* string through spaCy again because merging can creat new sentence boundaries.

                doc = nlp(merged_text)
                for new_sent in doc.sents:
                    cleanData = new_sent.text.strip()
                    if cleanData:
                        repaired_data.append({
                            "page": page_number,
                            "sentence": cleanData,
                            "header": merged_header
                        })
            else:
                # If no merge, process the sentence normally
                repaired_data.append({
                    "page": page_number,
                    "sentence": current["sentence"],
                    "header": current["header"]
                })
                i += 1
        return repaired_data


def split_into_sentences(pages):

    data = []
    for page in pages:

        page_number = page["page"]
        body_text_size = page["body_text_size"]

        """ We need the file path to open the page safely for OCR."""

        file_path = page.get("file_path")

        page_data = []   # collecting sentences for the current page
        previousBlock_y1 = None
        nextBlock_y0 = None
        current_header = None

        # Open the document briefly for this page's OCR
        with fitz.open(file_path) as tmp_doc:

            blocks = page["blocks"]
            for i, (block_text, block_font_size, x0, y0, x1, y1, is_bold) in enumerate(blocks):

                # HEADER DETECTION
                # consider space before and after header
                previousBlock_y1 = blocks[i-1][5] if i > 0 else None
                nextBlock_y0 = blocks[i+1][3] if i < len(blocks)-1 else None

                is_formula = is_formula_block(block_text, block_font_size, body_text_size)
                header = detect_heading(block_text, block_font_size, body_text_size, y0, y1, previousBlock_y1, nextBlock_y0, is_bold, is_formula)

                if header:
                    current_header = header
                    page_data.append({
                        "sentence": block_text.strip(),
                        "header": None,
                        "is_header": True
                    })

                    previousBlock_y1 = y1
                    continue

                # # skip formula blocks
                if is_formula_block(block_text, block_font_size, body_text_size):
                    continue

                # CAPTION AND FOOTER FILTERING
                # if the font size is smaller
                if block_font_size < (body_text_size - 1.1):
                    continue
                # If it starts with Figure, ...
                if re.match(r"^\s*(Figure|Fig\.|Table)\s*\d+", block_text, re.IGNORECASE):
                    continue
                # Algorithms & Pseudo-code (Input/Output/Line Numbers)
                if re.search(r"^\s*(Algorithm|Input:|Output:|Require:|Ensure:)", block_text, re.IGNORECASE) or \
                   re.match(r"^\s*\d+:", block_text) or \
                   re.search(r"^\s*(end if|end function|end for|return)\b", block_text, re.IGNORECASE):
                   #    "←" in block_text: # The assignment arrow is a dead giveaway
                    continue

                # CONTENT PROCESSING
                try:
                    cleanText = fix_hyphenation(block_text)

                    if not cleanText or cleanText.isspace():
                        previousBlock_y1 = y1
                        continue

                    doc = nlp(cleanText)

                    # 2. Check and print the state of the 'doc' object after spaCy call
                    if doc is None:
                        print("CRITICAL DEBUG: 'doc' object is None after calling nlp(). Skipping block.")
                        previousBlock_y1 = y1
                        continue

                    if doc.sents is None:
                        print("CRITICAL DEBUG: 'doc.sents' is None. Skipping block.")
                        previousBlock_y1 = y1
                        continue

                    for sent in doc.sents:

                        if sent is None:
                            print("CRITICAL DEBUG: 'sent' yielded None inside loop.")
                            continue

                        cleanSentence = sent.text.strip()
                        if cleanSentence:
                            page_data.append({
                                "sentence": cleanSentence,
                                "header": current_header,
                                "is_header": False,
                                "is_formula": False
                            })
                    previousBlock_y1 = y1

                except Exception as e:
                    print(f"Page {page_number} block error: {e}")
                    continue

        if page_data:
            repaired_data = repair_data(page_data, page_number)
            # Adding the repaired sentences to the final output list
            data.extend(repaired_data)

    return data