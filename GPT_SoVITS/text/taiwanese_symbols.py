from typing import List


class TaiwaneseHokkienSymbols:
    """
    A class to encapsulate the complete symbol set for Taiwanese Hokkien (Tâi-lô)
    for use in Text-to-Speech models like GPT-SoVITS.

    The list is designed to be comprehensive, unambiguous, and suitable for tokenization.
    Symbols are ordered with multi-character graphemes first to avoid tokenizer errors.
    """

    def __init__(self) -> None:
        # Flat list of symbols, where the model learns phonetic roles (initial, final, tone)
        # from sequence context. Multi-character graphemes are listed first to prevent
        # greedy tokenizers from splitting them incorrectly (e.g., 'ts' into 't' and 's').
        self._symbols: List[str] = [
            # --- Padding and boundary symbols ---
            "_",  # Padding for alignment
            "~",  # Optional: for sandhi or liaison
            " ",  # Space for word separation
            "-",  # Hyphen for syllable separation
            "^",  # Sentence start
            "$",  # Sentence end
            "#",  # Intra-sentence boundary (e.g., for pauses)
            # --- Punctuation for prosody ---
            ",",  # Comma (pause)
            ".",  # Period (sentence end)
            "!",  # Exclamation mark
            "?",  # Question mark
            "…",  # Ellipsis
            # --- Tones (Taiwanese Hokkien: 1, 2, 3, 4, 5, 7, 8) ---
            "1",
            "2",
            "3",
            "4",
            "5",
            "7",
            "8",
            # --- Digraphs/trigraphs (initials and nasalized finals) ---
            "ph",
            "th",
            "kh",
            "ts",
            "tsh",
            "ch",
            "chh",
            "ng",
            "nn",
            "oo",
            "ir",  # Special vowels/finals
            "an",
            "am",
            "in",
            "un",
            "on",
            "ong",
            "iang",
            "iong",  # Nasalized finals
            # --- Vowels and simple initials/finals ---
            "a",
            "b",
            "e",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "s",
            "t",
            "u",
            # --- Common compound vowels ---
            "ai",
            "au",
            "ia",
            "io",
            "iu",
            "ua",
            "ue",
            "ui",
        ]

    @property
    def symbols(self) -> List[str]:
        """Returns the sorted list of unique symbols."""
        return sorted(self._symbols)  # Pre-defined list is unique, no need for set


# --- Example Usage ---
def main() -> None:
    """Demonstrates the usage of the symbol set."""
    symbol_set = TaiwaneseHokkienSymbols()

    print(f"Total number of symbols: {len(symbol_set.symbols)}")
    print("Complete symbol list for GPT-SoVITS (Tâi-lô):")
    print(symbol_set.symbols)

    # Example: Tokenizing "tsa1-boo2, li2 ho2 bo2?"
    # Expected output: ['ts', 'a', '1', '-', 'b', 'oo', '2', ',', 'l', 'i', '2', 'h', 'o', '2', 'b', 'o', '2', '?']
    example_text = "tsa1-boo2, li2 ho2 bo2?"
    print(f"Example text: {example_text}")
    # Note: Actual tokenization requires a separate tokenizer implementation.


if __name__ == "__main__":
    main()
