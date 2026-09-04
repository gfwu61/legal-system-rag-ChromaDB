import re
from typing import List, Dict


def normalize_text(text: str) -> str:
    """
    Normalize text by fixing common OCR and PDF whitespace issues.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ").replace("\u3000", " ")
    return text


"""
The split_paragraphs function parses a single German legal text (such as the BGB) and breaks it down into individual section entries based on section headers (§).

How It Works

Text Normalization: Passes the input text through a helper function normalize_text.

Pattern Matching (Regex): Searches for section header lines formatted as § <number> <title> (e.g., § 535 Mietvertrag).

Title Handling: Checks if text immediately following the section number starts with a parenthesis (e.g., (1)). If so, it treats that text as body content rather than a title.

Fallback Mode: If no section symbol (§) is detected in the document, it captures the entire text block as a single default entry.

Return Value

The function returns a list of dictionaries (List[Dict]). Each dictionary represents an individual legal section with the following structure:

paragraph (str): The section number (e.g., "535" or "823a"). Returns "Unbekannt" in fallback mode.
title (str): The title/heading of the section (e.g., "Inhalt und Hauptpflichten des Mietvertrags"). Returns an empty string "" if untitled, or "Gesetzestext" in fallback mode.
content (str): The body text of the legal section.

Example Output Item:
[
    {
        "paragraph": "535",
        "title": "Inhalt und Hauptpflichten des Mietvertrags",
        "content": "(1) Durch den Mietvertrag wird der Vermieter verpflichtet, dem Mieter den Gebrauch der Mietsache während der Mietzeit zu gewähren..."
    },
    {
        "paragraph": "536",
        "title": "Mietminderung bei Sach- und Rechtsmängeln",
        "content": "(1) Hat die Mietsache zur Zeit der Überlassung an den Mieter einen Mangel..."
    }
]
"""
def split_paragraphs(full_text: str) -> List[Dict]:
    """
    Split a German legal document into paragraphs (§).

    Returns a list of dictionaries containing:
        - paragraph: paragraph number
        - title: paragraph title
        - content: paragraph text
    """
    full_text = normalize_text(full_text)

    # Match paragraph headers such as:
    # § 535 Mietvertrag
    paragraph_pattern = re.compile(
        r"(?m)^\s*§\s*(\d+[a-zA-Z]*)[ \t]+([A-ZÄÖÜ\(][^\n]*)$"
    )

    matches = list(paragraph_pattern.finditer(full_text))
    results = []

    # If no paragraph header is found, return the complete text
    if not matches:
        return [{
            "paragraph": "Unbekannt",
            "title": "Gesetzestext",
            "content": full_text.strip()
        }]

    for i, match in enumerate(matches):
        paragraph_number = str(match.group(1)).strip()
        paragraph_title = match.group(2)

        # Some paragraphs have no title and start directly with "(1)"
        if paragraph_title and paragraph_title.strip().startswith("("):
            paragraph_title = ""
            start_content = match.end(1)
        else:
            paragraph_title = paragraph_title.strip() if paragraph_title else ""
            start_content = match.end()

        # Determine the end position of the current paragraph
        if i + 1 < len(matches):
            end_content = matches[i + 1].start()
        else:
            end_content = len(full_text)

        paragraph_text = full_text[start_content:end_content].strip()

        results.append({
            "paragraph": paragraph_number,
            "title": paragraph_title,
            "content": paragraph_text
        })

    return results


"""
original text:§ 573 Ordentliche Kündigung des Vermieters
(1) Der Vermieter kann nur kündigen, wenn er ein berechtigtes Interesse an der Beendigung des Mietverhältnisses hat. Die Kündigung zum Zwecke der Mieterhöhung ist ausgeschlossen.
(2) Ein berechtigtes Interesse des Vermieters an der Beendigung des Mietverhältnisses liegt insbesondere vor, wenn

1.
    der Mieter seine vertraglichen Pflichten schuldhaft nicht unerheblich verletzt hat,
2.
    der Vermieter die Räume als Wohnung für sich, seine Familienangehörigen oder Angehörige seines Haushalts benötigt oder
3.
    der Vermieter durch die Fortsetzung des Mietverhältnisses an einer angemessenen wirtschaftlichen Verwertung des Grundstücks gehindert und dadurch erhebliche Nachteile erleiden würde; die Möglichkeit, durch eine anderweitige Vermietung als Wohnraum eine höhere Miete zu erzielen, bleibt außer Betracht; der Vermieter kann sich auch nicht darauf berufen, dass er die Mieträume im Zusammenhang mit einer beabsichtigten oder nach Überlassung an den Mieter erfolgten Begründung von Wohnungseigentum veräußern will.

(3) Die Gründe für ein berechtigtes Interesse des Vermieters sind in dem Kündigungsschreiben anzugeben. Andere Gründe werden nur berücksichtigt, soweit sie nachträglich entstanden sind.
(4) Eine zum Nachteil des Mieters abweichende Vereinbarung ist unwirksam.
"""


"""
The split_absaetze function takes the text content of a single legal section and splits it into its individual subsections (Absätze), identified by parenthesized numbers (e.g., (1), (2)).

How It Works
Pattern Matching (Regex): Scans for lines starting with parenthesized digits like (1) or (2).
Slicing: Extracts each subsection's full block, including any sub-lists or numbered points (e.g., 1., 2.) attached to that subsection.
Fallback Mode: If no parenthesized numbers are found, it treats the entire text as subsection "1".

Return Value
The function returns a list of dictionaries (List[Dict]). Each dictionary represents an individual subsection with the following structure:
absatz (str): The subsection number (e.g., "1", "2").
content (str): The complete text of that subsection, including the marker and any nested sub-items (Nummern).((1)absatz text\n\n + 1.\nalle nummer text)

[
    {
        "absatz": "1",
        "content": "(1) Der Vermieter kann nur kündigen, wenn er ein berechtigtes Interesse an der Beendigung des Mietverhältnisses hat. Die Kündigung zum Zwecke der Mieterhöhung ist ausgeschlossen."
    },
    {
        "absatz": "2",
        "content": "(2) Ein berechtigtes Interesse des Vermieters an der Beendigung des Mietverhältnisses liegt insbesondere vor, wenn\n\n1.\n    der Mieter seine vertraglichen Pflichten schuldhaft nicht unerheblich verletzt hat,\n2.\n    der Vermieter die Räume als Wohnung für sich, seine Familienangehörigen oder Angehörige seines Haushalts benötigt oder\n3.\n    der Vermieter durch die Fortsetzung des Mietverhältnisses an einer angemessenen wirtschaftlichen Verwertung des Grundstücks gehindert und dadurch erhebliche Nachteile erleiden würde; die Möglichkeit, durch eine anderweitige Vermietung als Wohnraum eine höhere Miete zu erzielen, bleibt außer Betracht; der Vermieter kann sich auch nicht darauf berufen, dass er die Mieträume im Zusammenhang mit einer beabsichtigten oder nach Überlassung an den Mieter erfolgten Begründung von Wohnungseigentum veräußern will."
    },
    {
        "absatz": "3",
        "content": "(3) Die Gründe für ein berechtigtes Interesse des Vermieters sind in dem Kündigungsschreiben anzugeben. Andere Gründe werden nur berücksichtigt, soweit sie nachträglich entstanden sind."
    },
    {
        "absatz": "4",
        "content": "(4) Eine zum Nachteil des Mieters abweichende Vereinbarung ist unwirksam."
    }
]

"""
# paragraph_text: = "content": paragraph_text from content of return of split_paragraphs()
def split_absaetze(paragraph_text: str) -> List[Dict]:
    """
    Split a paragraph into subsections (Absätze).

    Example:
        (1) ...
        (2) ...

    Returns a list of dictionaries containing:
        - absatz: subsection number
        - content: subsection text (absatz text + alle nummer text)
    """
    section_pattern = re.compile(r"(?m)^\s*\((\d+)\)")
    matches = list(section_pattern.finditer(paragraph_text))

    # If no subsection exists, treat the entire text as subsection 1
    if not matches:
        return [{
            "absatz": "1",
            "content": paragraph_text.strip()
        }]

    absaetze = []

    for i, match in enumerate(matches):
        absatz_number = str(match.group(1)).strip()
        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(paragraph_text)

        content = paragraph_text[start:end].strip()
    # content: /n/n or /n is from original paragraph-content, not inserted extra in code
        absaetze.append({
            "absatz": absatz_number,
            "content": content
        })

    return absaetze


# absatz_text: = "content": content
# contain all the nr's
#
"""
Function Description
The split_nummern function parses a structured legal text string (absatz_text) and extracts standalone numbered items (e.g., 1., 2.). It identifies any intro text preceding the first numbered item and attaches this intro to every individual item's content field.

Input

absatz_text (str): A string containing a subsection of legal text. It contain introductory text of subsection followed by numbered list items that sit on their own line.

Output

List[Dict]: A list of dictionaries. Returns an empty list if no numbered pattern match is found. Otherwise, each dictionary contains three key-value pairs:

"nummer" (str): The extracted number string (e.g., "1").

"content" (str): The full text snippet for this item, including the prepended intro_text (Absatz) followed by the item's text.

"intro_isolated" (str): The intro text that preceded the first numbered list item (subsection= Absatz), example:
(2) Ein berechtigtes Interesse des Vermieters an der Beendigung des Mietverhältnisses liegt insbesondere vor, wenn ... (this is Absatz)
"""


def split_nummern(absatz_text: str) -> List[Dict]:
    """Split a subsection into numbered legal items.

    Numbered items must occupy their own line, for example:

        1.
            der Mieter ...
        2.
            der Vermieter ...

    This prevents expressions such as "15. eines Monats"
    from being interpreted as legal item numbers.
    """
    # Regex pattern: Match digits followed by a dot that stand alone on a line
    # (?m)^ matches start of a line; \s* allows leading whitespace; \d+ captures numbers
    nummer_pattern = re.compile(r"(?m)^\s*(\d+)\.\s*$")

    # Find all regex matches within the input text
    matches = list(nummer_pattern.finditer(absatz_text))

    # If no standalone numbered list items exist, return an empty list
    if not matches:
        return []

    # Extract all text before the very first numbered match as the shared introductory text
    intro_text = absatz_text[: matches[0].start()].strip()

    results = []

    # Iterate through each matched number to slice its corresponding text block
    for i, match in enumerate(matches):
        nummer = match.group(1)
        start = match.start()

        # Set the end offset to the start of the next number match, or the end of the text if it's the last item
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(absatz_text)

        # Extract and clean the specific text segment for this list item
        nummer_text = absatz_text[start:end].strip()

        # Append the structured record to the results list
        results.append({
            "nummer": nummer,
            "content": nummer_text,
            "subsection_text": intro_text
        })

    return results




def extract_references(text: str) -> List[str]:
    """
    Extract all legal paragraph references from the text.

    Example:
        § 535
        §§ 535, 536

    Returns a list of unique paragraph numbers.
    """
    refs = re.findall(r"§+\s*(\d+[a-zA-Z]*)", text)
    return list(set([str(r).strip() for r in refs]))


    