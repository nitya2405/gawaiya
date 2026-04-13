from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    unk_token: str = "__UNK__"

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, value: str | None) -> int:
        if not value:
            return self.stoi[self.unk_token]
        return int(self.stoi.get(str(value), self.stoi[self.unk_token]))

    def decode(self, idx: int) -> str:
        return self.itos[int(idx)]

    def to_json(self) -> dict:
        return {"itos": self.itos, "unk_token": self.unk_token}

    @staticmethod
    def from_json(obj: dict) -> "Vocab":
        itos = list(obj["itos"])
        unk_token = str(obj.get("unk_token", "__UNK__"))
        stoi = {s: i for i, s in enumerate(itos)}
        if unk_token not in stoi:
            stoi[unk_token] = len(itos)
            itos.append(unk_token)
        return Vocab(stoi=stoi, itos=itos, unk_token=unk_token)


def build_vocab(values: list[str], *, unk_token: str = "__UNK__") -> Vocab:
    uniq = sorted({str(v) for v in values if v is not None})
    itos = [unk_token, *uniq]
    stoi = {s: i for i, s in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, unk_token=unk_token)


def save_vocab(path: str | Path, vocab: Vocab) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab.to_json(), f, ensure_ascii=False, indent=2)


def load_vocab(path: str | Path) -> Vocab:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return Vocab.from_json(obj)

