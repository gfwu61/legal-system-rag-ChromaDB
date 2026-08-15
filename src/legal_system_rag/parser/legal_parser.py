import re
from typing import List, Dict


def normalize_text(text: str) -> str:
    """
    Normalize text by fixing common OCR and PDF whitespace issues.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ").replace("\u3000", " ")
    return text


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
        r"(?m)^\s*§\s*(\d+[a-zA-Z]*)[ \t]+([A-ZÄÖÜ][^\n]*)$"
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


def split_absaetze(paragraph_text: str) -> List[Dict]:
    """
    Split a paragraph into subsections (Absätze).

    Example:
        (1) ...
        (2) ...

    Returns a list of dictionaries containing:
        - absatz: subsection number
        - content: subsection text
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

        absaetze.append({
            "absatz": absatz_number,
            "content": content
        })

    return absaetze

def split_nummern(absatz_text: str) -> List[Dict]:
    """
    Split a subsection into numbered legal items.

    Numbered items must occupy their own line, for example:

        1.
            der Mieter ...
        2.
            der Vermieter ...

    This prevents expressions such as "15. eines Monats"
    from being interpreted as legal item numbers.
    """
    nummer_pattern = re.compile(
        r"^\s*(\d+)\.\s*$",
        re.MULTILINE,
    )

    matches = list(nummer_pattern.finditer(absatz_text))

    if not matches:
        return []

    intro_text = absatz_text[:matches[0].start()].strip()

    results = []

    for i, match in enumerate(matches):
        nummer = match.group(1)
        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(absatz_text)

        nummer_text = absatz_text[start:end].strip()

        full_content = (
            f"{intro_text}\n{nummer_text}"
            if intro_text
            else nummer_text
        )

        results.append({
            "nummer": nummer,
            "content": full_content,
            "intro_isolated": intro_text,
        })

    return results
    

def split_nummern_old(absatz_text: str) -> List[Dict]:
    """
    Split a subsection into numbered items.

    Example:
        1. ...
        2. ...
        3. ...

    Date expressions such as "1. Januar" are ignored.
    """
    nummer_pattern = re.compile(
        r"(\d+)\.(?!\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember))\s*"
    )

    matches = list(nummer_pattern.finditer(absatz_text))

    if not matches:
        return []

    # Text before the first numbered item.
    # This introductory text is attached to every numbered item.
    intro_text = absatz_text[:matches[0].start()].strip()

    results = []

    for i, match in enumerate(matches):
        nummer = str(match.group(1)).strip()
        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(absatz_text)

        nummer_text = absatz_text[start:end].strip()

        # Preserve the introductory text for each numbered item
        full_content = (
            f"{intro_text}\n{nummer_text}"
            if intro_text
            else nummer_text
        )

        results.append({
            "nummer": nummer,
            "content": full_content,
            "intro_isolated": intro_text
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


    