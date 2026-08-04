import re
from typing import List, Union

import numpy as np

VOICE_CHANGE_TOKEN = "<COC>"
STEP_CHANGE_TOKEN = "<COR>"

M2S_STREAM_KEYS = [
    "offset", "downbeat", "duration", "pitch", "accidental", "keysignature",
    "velocity", "grace", "trill", "staccato", "voice", "stem", "hand",
]

_KERN_CLEF_MAP = {
    "clef_treble": "*clefG2",
    "clef_treblebar": "*clefG2",
    "clef_bass": "*clefF4",
    "clef_alto": "*clefC3",
    "clef_tenor": "*clefC4",
    "clef_soprano": "*clefC1",
    "clef_mezzosoprano": "*clefC2",
    "clef_baritone": "*clefF3",
    "clef_subbass": "*clefF5",
}

_KERN_NOTE_NAMES = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]


def _quarter_to_kern_duration(q: float) -> str:
    """Convert a duration given in quarter-note units to a kern duration token.

    The nearest value among kern classes 1..128 with up to 3 augmentation dots
    is selected (ST+/MIDI2Score durations that come from tuplets are therefore
    approximated, which is acceptable for MV2H note evaluation).
    """
    q = max(float(q), 1.0 / 128.0)
    best = None
    best_diff = float("inf")
    for e in range(0, 8):
        k = 2 ** e
        for d in range(0, 4):
            value = (4.0 / k) * (2.0 - 1.0 / (2 ** d))
            diff = abs(value - q)
            if diff < best_diff:
                best_diff = diff
                best = (k, d)
    k, d = best
    return str(k) + "." * d


def _kern_pitch_marks(octave: int) -> int:
    """Number of octave marks for a **kern pitch at the given octave."""
    if octave >= 5:
        return octave - 4
    if octave == 4:
        return 0
    return 4 - octave + 1


def _st_plus_pitch_to_kern(pitch_name: str) -> Union[str, None]:
    """Convert an ST+ pitch token (e.g. 'C4', 'F#4', 'Bb3') to its **kern form."""
    m = re.match(r"([A-G])([#b]*)([0-9]+)", pitch_name)
    if not m:
        return None
    letter, acc, octave = m.group(1), m.group(2), int(m.group(3))
    base = letter.lower()
    marks = _kern_pitch_marks(octave)
    if octave >= 5:
        kern_letter = base.upper() * marks
    elif octave == 4:
        kern_letter = base
    else:
        kern_letter = base * marks
    kern_acc = {"#": "#", "b": "-"}.get(acc, "")
    return kern_letter + kern_acc


def _midi_to_kern(midi: int) -> str:
    """Convert a MIDI note number to its **kern form (C4 = MIDI 60 = 'c')."""
    midi = max(0, min(127, int(midi)))
    base = _KERN_NOTE_NAMES[midi % 12]
    acc = "#" if len(base) > 1 else ""
    octave = midi // 12 - 1
    marks = _kern_pitch_marks(octave)
    if octave >= 5:
        kern_letter = base[0].upper() * marks
    elif octave == 4:
        kern_letter = base[0]
    else:
        kern_letter = base[0] * marks
    return kern_letter + acc


def _key_to_kern(key_token: str) -> str:
    m = re.match(r"key_(sharp|flat)_([0-9]+)", key_token)
    if not m:
        return "*k[]"
    kind, n = m.group(1), int(m.group(2))
    n = max(0, min(7, n))
    if n == 0:
        return "*k[]"
    if kind == "sharp":
        return "*k[" + "".join(x + "#" for x in "fcgdaeb"[:n]) + "]"
    return "*k[" + "".join(x + "-" for x in "beadgcf"[:n]) + "]"


def _spines_to_stream(spines: List[List[str]]) -> List[str]:
    """Interleave spines row by row (DOT-padded) into the flat AR token stream."""
    if not spines:
        return [STEP_CHANGE_TOKEN]
    max_rows = max(len(s) for s in spines)
    out = []
    for r in range(max_rows):
        for s in spines:
            out.append(s[r] if r < len(s) else "DOT")
        out.append(STEP_CHANGE_TOKEN)
    return out


def st_plus_to_kern(st_plus_tokens: List[str]) -> List[str]:
    """Convert a flat ST+ token stream into a row-major **kern token stream.

    Notes/rests are grouped per hand (R/L) into separate spines; bars, clefs,
    key signatures and time signatures are mapped to their **kern equivalents.
    The output is compatible with the MV2H token-stream format (rows of tokens
    delimited by <COR>).
    """
    spines = {}
    hand_order = []
    current_hand = None
    pending = None  # ("rest" | pitch name, quarters)

    def get_spine(hand):
        if hand not in spines:
            spines[hand] = []
            hand_order.append(hand)
        return spines[hand]

    def flush_pending():
        nonlocal pending
        if pending is None:
            return
        token, q = pending
        dur = _quarter_to_kern_duration(q)
        if token == "rest":
            get_spine(current_hand).append(dur + "r")
        else:
            kern_pitch = _st_plus_pitch_to_kern(token)
            if kern_pitch is not None:
                get_spine(current_hand).append(dur + kern_pitch)
        pending = None

    for tok in st_plus_tokens:
        if not isinstance(tok, str) or not tok:
            continue
        base, sep, suffix = tok.rpartition("bar")
        if sep and base:
            tok = base
            bar = True
        else:
            bar = False

        if tok == "R":
            current_hand = "R"
        elif tok == "L":
            current_hand = "L"
        elif tok == "<voice>":
            pass
        elif tok == "</voice>":
            flush_pending()
        elif tok == "rest":
            pending = ("rest", None)
        elif tok.startswith("note_"):
            pending = (tok[5:], None)
        elif tok.startswith("len_"):
            frac = tok[4:]
            if "/" in frac:
                num, den = frac.split("/")
                try:
                    q = float(num) / float(den)
                except ValueError:
                    q = 0.0
            else:
                try:
                    q = float(frac)
                except ValueError:
                    q = 0.0
            pending = (pending[0], q) if pending else None
            flush_pending()
        elif tok == "bar":
            for spine in spines.values():
                spine.append("=")
        elif tok.startswith("clef_"):
            clef = _KERN_CLEF_MAP.get(tok)
            if clef is not None:
                for spine in spines.values():
                    spine.append(clef)
        elif tok.startswith("key_"):
            for spine in spines.values():
                spine.append(_key_to_kern(tok))
        elif tok.startswith("time_"):
            try:
                num, den = tok[5:].split("/")
                for spine in spines.values():
                    spine.append(f"*M{num}/{den}")
            except ValueError:
                pass
        # stem_*, beam_*, tie_*, slur_*, accent, staccato, tenuto, chord_*, R, L... skipped

        if bar:
            for spine in spines.values():
                spine.append("=")

    if pending is not None:
        flush_pending()

    return _spines_to_stream([spines[h] for h in hand_order])


def midi2score_to_tokens(streams: dict) -> List[str]:
    """Tokenize a MIDI2Score streams dict into the flat AR token stream.

    streams values are lists of token ids (argmax of the one-hot buckets),
    following M2S_STREAM_KEYS order (pad excluded).
    """
    if not streams:
        return []
    n_rows = min(len(streams[k]) for k in M2S_STREAM_KEYS if k in streams)
    tokens = []
    for i in range(n_rows):
        for key in M2S_STREAM_KEYS:
            tokens.append(f"{key}_{streams[key][i]}")
    return tokens


def midi2score_to_kern(tokens: List[str]) -> List[str]:
    """Convert a flat MIDI2Score token stream into a row-major **kern stream.

    Notes are reassembled from the per-note streams (in token order) and grouped
    per hand into separate spines.
    """
    notes = []
    cur = {}
    for tok in tokens:
        stream, sep, val = tok.rpartition("_")
        if not sep or stream not in M2S_STREAM_KEYS:
            continue
        try:
            val = int(val)
        except ValueError:
            continue
        cur[stream] = val
        if stream == "hand":
            notes.append(cur)
            cur = {}

    spines = {}
    hand_order = []
    for note in notes:
        hand = note.get("hand", 2)
        if hand not in spines:
            spines[hand] = []
            hand_order.append(hand)
        q = note.get("duration", 0) / 24.0
        dur = _quarter_to_kern_duration(q)
        midi = note.get("pitch", 60)
        spines[hand].append(dur + _midi_to_kern(midi))

    return _spines_to_stream([spines[h] for h in hand_order])


class krnParser:
    """Main Kern parser operations class."""

    def __init__(self, use_voice_change_token: bool = True) -> None:
        self.reserved_words = ["clef", "k[", "*M"]
        self.reserved_dot = "."
        self.reserved_dot_EncodedCharacter = "DOT"
        self.clef_change_other_voices = "*"
        self.comment_symbols = ["*", "!"]
        self.voice_change = VOICE_CHANGE_TOKEN  # change-of-column (coc) token
        self.step_change = STEP_CHANGE_TOKEN  # change-of-row (cor) token
        self.use_voice_change_token = use_voice_change_token

    # ---------------------------------------------------------------------------- AUXILIARY FUNCTIONS

    def _readSrcFile(self, text: str) -> np.ndarray:
        """Adequate a Kern file content to the correct format for further processes."""
        in_src = text.splitlines()

        # Locating line with the headers
        it_headers = 0
        while "**kern" not in in_src[it_headers]:
            it_headers += 1
        header_fields = in_src[it_headers].split("\t")
        n_kern_cols = int(np.sum(np.array(header_fields) == "**kern"))
        has_dynam = any(field == "**dynam" for field in header_fields)

        # Locating lines with comments (to be removed)
        in_src_nocomments = []
        for line in in_src:
            if not line.strip().startswith("!"):
                fields = line.split("\t")
                # ASAP piano kern carries a trailing **dynam spine: drop it.
                # Extra columns added by spine splits (^) on kern voices are kept.
                if has_dynam and len(fields) > n_kern_cols:
                    fields = fields[:-1]
                in_src_nocomments.append(fields)

        # Rows may have different numbers of columns (spine splits/merges),
        # so pad the shorter ones to build a rectangular matrix. "!" cells are
        # dropped later by cleanKernToken (they are comments).
        max_width = max(len(fields) for fields in in_src_nocomments)
        out_src = np.array(
            [fields + ["!"] * (max_width - len(fields)) for fields in in_src_nocomments]
        )

        return out_src

    def _postprocessKernSequence(self, in_score: np.ndarray) -> np.ndarray:
        """Exchanging '*' for the actual symbol."""

        # Retrieving positions with '*'
        positions = sorted(list(set(np.where(in_score == "*")[1])))

        # For each position,
        # we retrieve the last explicit clef symbol and include it in the stream
        for single_position in positions:
            for it_voice in range(in_score.shape[0]):
                if in_score[it_voice, single_position] == "*":
                    clef_positions = np.where(
                        np.char.startswith(in_score[it_voice], "*clef")
                    )[0]
                    if len(clef_positions) > 0:
                        new_element = in_score[it_voice, max(clef_positions)]
                    else:
                        new_element = "*"
                    in_score[it_voice, single_position] = new_element
                pass
            pass
        pass

        return in_score

    def cleanKernFile(self, text: str) -> np.ndarray:
        """Convert complete kern sequence to CLEAN kern format.

        Returns a [voices, rows] matrix. Tokens dropped by cleanKernToken are
        replaced with "!" so every voice keeps its row alignment even when the
        file has spine splits (variable number of columns per row).
        """
        in_file = self._readSrcFile(text=text)

        cleaned = in_file.astype(str)
        for it_row in range(in_file.shape[0]):
            for it_voice in range(in_file.shape[1]):
                token = self.cleanKernToken(in_file[it_row, it_voice])
                cleaned[it_row, it_voice] = token if token is not None else "!"

        # Processing individual voices (postprocess expects [voices, rows])
        out_score = self._postprocessKernSequence(cleaned.T)

        return out_score

    def cleanKernToken(self, in_token: str) -> Union[str, None]:
        """Convert a kern token to its CLEAN equivalent."""
        out_token = None  # Default

        if any([u in in_token for u in self.reserved_words]):  # Relevant reserved tokens
            out_token = in_token

        elif in_token == self.reserved_dot:  # Case when using "." for sync. voices
            out_token = self.reserved_dot_EncodedCharacter

        elif in_token.strip() == self.clef_change_other_voices:  # Clef change in other voices
            out_token = in_token

        elif any([in_token.startswith(u) for u in self.comment_symbols]):  # Comments
            out_token = None

        elif in_token.startswith("s"):  # Slurs
            out_token = "s"

        elif "=" in in_token:  # Bar lines
            out_token = "="

        elif not "q" in in_token:
            if "rr" in in_token:  # Multirest
                out_token = re.findall(r"rr[0-9]+", in_token)[0]
            elif "r" in in_token:  # Rest
                out_token = in_token.split("r")[0] + "r"
            else:  # Music note
                note = re.findall(r"\[*\d+[.]*[a-gA-G]+[n#-]*\]*", in_token)
                if not note:
                    # Bare pitch (duration omitted: same as the previous note)
                    note = re.findall(r"[a-gA-G]+[n#-]*", in_token)
                out_token = note[0] if note else None  # e.g. "p" (pedal) is dropped

        return out_token

    # ---------------------------------------------------------------------------- CONVERT CALL

    def convert(self, text: str) -> List[str]:
        out = self.cleanKernFile(text).T

        out_line = []
        for t in out:
            if all(v == "!" for v in t):
                continue
            for v in t:
                # Missing voices are kept as DOT so every row has the same
                # number of voices (required by the MV2H evaluation format).
                if v == "!":
                    v = self.reserved_dot_EncodedCharacter
                out_line.append(str(v))
                if self.use_voice_change_token:
                    out_line.append(self.voice_change)
            if self.use_voice_change_token and out_line and out_line[-1] == self.voice_change:
                del out_line[-1]
            out_line.append(self.step_change)
        del out_line[-1]

        return out_line
