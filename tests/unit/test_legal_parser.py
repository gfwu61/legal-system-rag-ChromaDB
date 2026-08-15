from legal_system_rag.parser import split_absaetze, split_nummern


def test_573c_15_is_not_a_legal_number():
    text = """§ 573c Fristen der ordentlichen Kündigung
(1) Die Kündigung ist spätestens am dritten Werktag eines Kalendermonats zum Ablauf des übernächsten Monats zulässig. Die Kündigungsfrist für den Vermieter verlängert sich nach fünf und acht Jahren seit der Überlassung des Wohnraums um jeweils drei Monate.
(2) Bei Wohnraum, der nur zum vorübergehenden Gebrauch vermietet worden ist, kann eine kürzere Kündigungsfrist vereinbart werden.
(3) Bei Wohnraum nach § 549 Abs. 2 Nr. 2 ist die Kündigung spätestens am 15. eines Monats zum Ablauf dieses Monats zulässig.
(4) Eine zum Nachteil des Mieters von Absatz 1 oder 3 abweichende Vereinbarung ist unwirksam.
"""

    # Split § 573c into subsections
    absaetze = split_absaetze(text)

    assert len(absaetze) == 4

    assert absaetze[0]["absatz"] == "1"
    assert absaetze[1]["absatz"] == "2"
    assert absaetze[2]["absatz"] == "3"
    assert absaetze[3]["absatz"] == "4"

    # Check subsection 3
    absatz_3 = absaetze[2]

    nummern = split_nummern(absatz_3["content"])

    # "15. eines Monats" must NOT be interpreted as legal item 15
    assert nummern == []



def test_573_real_legal_numbers_are_detected():
    text = """(2) Ein berechtigtes Interesse des Vermieters an der Beendigung des Mietverhältnisses liegt insbesondere vor, wenn
1.
der Mieter seine vertraglichen Pflichten schuldhaft nicht unerheblich verletzt hat,

2.
der Vermieter die Räume als Wohnung für sich, seine Familienangehörigen oder Angehörige seines Haushalts benötigt oder

3.
der Vermieter durch die Fortsetzung des Mietverhältnisses an einer angemessenen wirtschaftlichen Verwertung des Grundstücks gehindert und dadurch erhebliche Nachteile erleiden würde.
"""

    nummern = split_nummern(text)

    assert len(nummern) == 3

    assert nummern[0]["nummer"] == "1"
    assert nummern[1]["nummer"] == "2"
    assert nummern[2]["nummer"] == "3"