from __future__ import annotations


class CharTokenizer:
    """Character-level tokenizer with a few multi-character special tokens."""

    def __init__(
        self,
        chars: str = (
            "0123456789+-*/=;()"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ),
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        extra_specials: list[str] | None = None,
        tokens: list[str] | None = None,
    ) -> None:
        specials = [pad_token, bos_token, eos_token] + list(extra_specials or [])
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.extra_specials = list(extra_specials or [])

        self.id_to_token = list(tokens) if tokens is not None else list(chars) + specials
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}
        self.vocab_size = len(self.id_to_token)

        self.pad_id = self.token_to_id[pad_token]
        self.bos_id = self.token_to_id[bos_token]
        self.eos_id = self.token_to_id[eos_token]

        # Longest-first so "</think>" wins over "<think>" / "<"
        self._multi_specials = sorted(
            [tok for tok in specials if len(tok) > 1],
            key=len,
            reverse=True,
        )
        self._skip_ids = {self.pad_id, self.bos_id, self.eos_id}

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for tok in self._multi_specials:
                if text.startswith(tok, i):
                    ids.append(self.token_to_id[tok])
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            if ch not in self.token_to_id:
                raise ValueError(f"Unknown character: {ch!r}")
            ids.append(self.token_to_id[ch])
            i += 1
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out: list[str] = []
        for i in ids:
            if skip_special and i in self._skip_ids:
                continue
            out.append(self.id_to_token[i])
        return "".join(out)

    def __len__(self) -> int:
        return self.vocab_size
