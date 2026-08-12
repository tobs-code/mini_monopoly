"""Tests für data/tokenizer.py (MiniTokenizer, §5.1/§5.1a).

Verifiziert:
- Vokabular-Größe (d_vocab = 48, PAD = ID 0).
- Digit-Encoding der Miete/Cash (Stellenwerte TH/H/T/O).
- Kontext-basierte Disambiguierung Würfel (ROLL) vs. Miete (PAY_RENT),
  insbesondere bei einstelligen/cash-gecappten Mietwerten.
- Dekodier-Round-Trip.
- Tokenisierbarkeit ALLER Spiele im vollen Datensatz (kein OOV, kein
  Kontext-Error) + Messung der tokenisierten Kontextlängen.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.tokenizer import MiniTokenizer, encode_int, TokenizerError
from data.splits import load_games

FIXTURE = os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl")


def test_vocab_size():
    t = MiniTokenizer()
    assert t.d_vocab == 48, f"d_vocab={t.d_vocab}, erwartet 48"
    assert t.pad_id == 0
    assert t.stoi["<PAD>"] == 0
    assert t.stoi["MONEY"] == 15
    assert len(t.itos) == len(set(t.itos)), "Vokabular enthält Duplikate"


def test_encode_int_place_values():
    # 750 -> TH0 H7 T5 O0
    assert encode_int(750) == ["MONEY", "TH0", "H7", "T5", "O0"]
    # 1500 -> TH1 H5 T0 O0
    assert encode_int(1500) == ["MONEY", "TH1", "H5", "T0", "O0"]
    # 900 -> TH0 H9 T0 O0  (Korrektheit gegen §5.1a-Doc-Beispiel, das H9 T3 O0=930 zeigte)
    assert encode_int(900) == ["MONEY", "TH0", "H9", "T0", "O0"]
    # 0 -> TH0 H0 T0 O0
    assert encode_int(0) == ["MONEY", "TH0", "H0", "T0", "O0"]
    # 25 -> TH0 H0 T2 O5
    assert encode_int(25) == ["MONEY", "TH0", "H0", "T2", "O5"]


def test_encode_int_range():
    import pytest
    t = MiniTokenizer()
    with pytest.raises(TokenizerError):
        encode_int(2000)
    with pytest.raises(TokenizerError):
        encode_int(-5)


def test_dice_vs_rent_disambiguation():
    t = MiniTokenizer()
    # ROLL -> Dice-Token (atomar), PAY_RENT -> Digit-Sequenz
    seq = ["START", "P0", "ROLL", "5", "P0", "PAY_RENT", "75"]
    ids = t.tokenize_events(seq)
    toks = t.decode_ids(ids)
    assert toks == ["START", "P0", "ROLL", "D5", "P0", "PAY_RENT",
                    "MONEY", "TH0", "H0", "T7", "O5"], toks


def test_cash_capped_single_digit_rent():
    # Miete kann cash-gecappt einstellig/0 sein -> muss trotzdem als Miete
    # (Digit-Sequenz), NICHT als Wuerfel erkannt werden.
    t = MiniTokenizer()
    for rent in (0, 5, 25, 50, 150):
        seq = ["P0", "PAY_RENT", str(rent)]
        toks = t.decode_ids(t.tokenize_events(seq))
        assert toks[0] == "P0" and toks[1] == "PAY_RENT", toks
        # letzte 5 Tokens = MONEY-Digit-Sequenz, kein D-Token
        assert toks[-5] == "MONEY", toks
        assert not toks[-1].startswith("D"), toks
        assert toks[-1] == f"O{rent % 10}", toks


def test_ambiguous_number_raises():
    t = MiniTokenizer()
    import pytest
    # numerisches Token ohne vorausgehendes ROLL/PAY_RENT -> Error, kein stilles Raten
    with pytest.raises(TokenizerError):
        t.tokenize_events(["P0", "5"])
    with pytest.raises(TokenizerError):
        t.tokenize_events(["START", "42"])


def test_unknown_token_raises():
    t = MiniTokenizer()
    import pytest
    with pytest.raises(TokenizerError):
        t.tokenize_events(["START", "FOOBAR"])


def test_roundtrip_and_full_dataset():
    """Alle Spiele im vollen Datensatz müssen tokenisierbar sein (kein OOV),
    und die Token-Zählung muss die erwartete Struktur liefern:
    - 0 Exceptions über den gesamten Korpus (Würfel immer 1..6, keine Ambiguity).
    - Genau 80000 D-Tokens (= 40 Turns x 2000 Spiele), alle D1..D6 > 0.
    - MONEY-Anzahl == Anzahl der PAY_RENT-Events (jede Miete genau 1 MONEY).
    - TH0 > 0 (Mieten <= 150 -> Tausenderstelle 0), TH1 == 0: in V0 wird nur
      die Miete (Cash-Fluss) tokenisiert, Cash selbst nie (History-only, §5.3);
      TH1 bleibt als bewusst reserviertes Token für die S3-Preiserweiterung (§5.6)."""
    from collections import Counter
    t = MiniTokenizer()
    games = load_games(FIXTURE)
    lengths = []
    counter = Counter()
    n_payrent = 0
    exceptions = 0
    for g in games:
        n_payrent += g["events"].count("PAY_RENT")
        try:
            ids = t.tokenize_events(g["events"])
        except Exception as e:
            exceptions += 1
            raise AssertionError(f"Tokenisierung fehlgeschlagen: {e}")
        assert ids, "leere Sequenz"
        assert all(0 <= i < t.d_vocab for i in ids)
        for i in ids:
            counter[t.itos[i]] += 1
        lengths.append(len(ids))
    n = len(games)
    print(f"\nTokenisierte Spiele: {n}, Exceptions: {exceptions}")
    assert exceptions == 0, "mind. eine Tokenisierung warf eine Exception"
    # D-Tokens
    d_total = sum(counter[f"D{d}"] for d in range(1, 7))
    assert d_total == 40 * n, f"D-Tokens={d_total}, erwartet {40*n}"
    for d in range(1, 7):
        assert counter[f"D{d}"] > 0, f"D{d} kommt nie vor"
    # MONEY == PAY_RENT, TH0 aktiv, TH1 reserviert (0 in V0)
    assert counter["MONEY"] == n_payrent, \
        f"MONEY={counter['MONEY']} != PAY_RENT={n_payrent}"
    assert counter["TH0"] == n_payrent, f"TH0={counter['TH0']}"
    assert counter["TH1"] == 0, f"TH1={counter['TH1']}, erwartet 0 (V0, nur Rent tokenisiert)"
    # Kontextlänge
    print(f"Kontextlänge: min={min(lengths)} p50={sorted(lengths)[n//2]} "
          f"max={max(lengths)}")
    print(f"Spiele mit Länge > 256: {sum(1 for l in lengths if l > 256)}")
    return max(lengths)


def test_alignment_raw_index():
    """tokenize_events_aligned: pro Output-Token der korrekte Roh-Event-Index.

    Ein Miet-Event ('75') expandiert auf 5 Tokens, alle mit demselben Roh-Index;
    Würfel- und Event-Tokens haben 1:1-Zuordnung.
    """
    t = MiniTokenizer()
    seq = ["START", "P0", "ROLL", "5", "P0", "PAY_RENT", "75"]
    ids, raw = t.tokenize_events_aligned(seq)
    assert len(ids) == len(raw)
    # Würfel '5' -> 1 Token, Roh-Index 3
    assert raw[3] == 3 and t.decode_ids([ids[3]]) == ["D5"]
    # PAY_RENT (Roh-Index 5) + 5 Digit-Tokens (Roh-Index 6)
    assert raw[4] == 4 and raw[5] == 5
    assert raw[6:] == [6] * 5, raw
    # Konsistenz mit tokenize_events
    assert ids == t.tokenize_events(seq)


if __name__ == "__main__":
    test_vocab_size()
    test_encode_int_place_values()
    test_encode_int_range()
    test_dice_vs_rent_disambiguation()
    test_cash_capped_single_digit_rent()
    test_ambiguous_number_raises()
    test_unknown_token_raises()
    test_alignment_raw_index()
    max_len = test_roundtrip_and_full_dataset()
    print("Alle Tokenizer-Tests bestanden.")
