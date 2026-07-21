"""Morse code encoding and blink-sequence decoding."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# International Morse code table (A–Z, 0–9, basic punctuation).
MORSE_TABLE: dict[str, str] = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
    ".-.-.-": ".",
    "--..--": ",",
    "..--..": "?",
    "-..-.": "/",
}

REVERSE_TABLE: dict[str, str] = {v: k for k, v in MORSE_TABLE.items()}


@dataclass
class MorseDecoder:
    """
    Accumulates dot/dash blinks and decodes them into text.

    Timing gaps (after the last blink) determine boundaries:
      - letter_gap: finish current letter
      - word_gap: finish letter and append a space
    """

    letter_gap: float = 1.2
    word_gap: float = 2.5

    current_pattern: str = field(default="", init=False)
    decoded_text: str = field(default="", init=False)
    last_event_time: float | None = field(default=None, init=False)

    def add_symbol(self, symbol: str) -> None:
        """Record a dot ('.') or dash ('-') from a blink."""
        self.current_pattern += symbol
        self.last_event_time = time.monotonic()

    def tick(self) -> str | None:
        """
        Check idle time and commit letter/word if a gap threshold is met.

        Returns the character that was decoded, or None if nothing changed.
        """
        if not self.current_pattern or self.last_event_time is None:
            return None

        idle = time.monotonic() - self.last_event_time
        if idle < self.letter_gap:
            return None

        pattern = self.current_pattern
        self.current_pattern = ""
        self.last_event_time = None

        if idle >= self.word_gap:
            char = MORSE_TABLE.get(pattern, "?")
            self.decoded_text += char + " "
            return char + " "

        char = MORSE_TABLE.get(pattern, "?")
        self.decoded_text += char
        return char

    def reset(self) -> None:
        self.current_pattern = ""
        self.decoded_text = ""
        self.last_event_time = None

    @staticmethod
    def pattern_for_char(char: str) -> str | None:
        return REVERSE_TABLE.get(char.upper())
