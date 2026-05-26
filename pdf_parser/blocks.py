import fitz # PyMuPDF
from statistics import median
import re


def extract_blocks(file_path):

    doc = fitz.open(file_path)
    pages = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        blocks = page.get_text("dict")["blocks"]

        block_data = []
        font_sizes = []
        x_positions = []

        for block in blocks:
            if block["type"] != 0:  # Skip non-text blocks
                continue

            # bounding box
            x0, y0, x1, y1 = block["bbox"]
            block_text = ""
            block_font_sizes = []
            is_bold = False

            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"]   # add text from this span
                    block_font_sizes.append(span["size"])  # collect the font size
                    x_positions.append(span["bbox"][0]) # position of this span (left)
                    is_bold = is_bold or bool(span["flags"] & 2**4)  # check if bold (assuming flag 1 indicates bold)

            if block_text.strip():
                avg_size = sum(block_font_sizes) / len(block_font_sizes)
                block_data.append((block_text.strip(), avg_size, x0, y0, x1, y1, is_bold))  # ("Introduction", 18.5, 120)
                font_sizes.extend(block_font_sizes)

        # If no text found on page, skip safely
        if not font_sizes:
            continue

        body_text_size = median(font_sizes)  # Most common font size -> body text size

        # JUST TEXT IN THE BOUNDARY
        # Identify real text blocks
        real_blocks = []
        for (text, size, x0, y0, x1, y1, is_bold) in block_data:
            if len(text) > 40:   # if line is longer than 40 characters, it's real text
                real_blocks.append((text, size, x0, y0, x1, y1, is_bold))

        # finding the left and right boundary of real text
        if real_blocks:
            min_x = min(b[2] for b in real_blocks)  # x0 left
            max_x = max(b[3] for b in real_blocks)  # x1 right
        else:
            min_x = min(x0 for (_, _, x0, _, _, _, _) in block_data)
            max_x = max(x1 for (_, _, _, _, x1, _, _) in block_data)

        # keep only blocks inside this region
        clean_block_data = [
            (text, size, x0, y0, x1, y1, is_bold)
            for (text, size, x0, y0, x1, y1, is_bold) in block_data
            if x0 >= min_x - 5 and x1 <= max_x + 5   # tiny tolerance
        ]


        final_blocks_for_processing = []

        for (text, size, x0, y0, x1, y1, is_bold) in clean_block_data:
            stripped_text = text.strip()

            if re.fullmatch(r"[\d\s\W]+", stripped_text):
                # To be extra safe, only filter if the block is very short (e.g., < 10 characters)
                if len(stripped_text) < 10:
                    continue

            final_blocks_for_processing.append((stripped_text, size, x0, y0, x1, y1, is_bold))

        pages.append({
            "page": page_number + 1,
            "blocks": final_blocks_for_processing,
            "body_text_size": body_text_size,
            "file_path": file_path,
            "page_index": page_number
        })


    if pages:
        print(f"DEBUG: extract_blocks returning list of length: {len(pages)}")
        print(f"DEBUG: Type of first item in pages_data: {type(pages[0])}")
    return doc, pages