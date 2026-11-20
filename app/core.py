"""Crop health sequences, frequent itemsets, and a small next-item language model.

Milestone 1 turns crop rotations and pest observations into baskets.
Milestone 2 mines them with thilimarketbasket.
Milestone 3 continued-pretrains a small causal model on those sequences.
Milestone 4 scores next-item predictions on held-out sequences.
Milestone 5 is the payload the CLI prints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from thilimarketbasket.mine import frequent_itemsets, rules

# Ordered notes from the field. The same list is a basket for mining (order ignored)
# and a sequence for next-item prediction (order kept).
PEST_SEQUENCES: list[list[str]] = [
    ["rice", "brown planthopper", "sheath blight", "drain the field"],
    ["tomato", "whitefly", "leaf curl", "remove infected plants"],
    ["wheat", "stripe rust", "aphids", "use resistant seed"],
    ["cotton", "aphids", "bollworm", "scout the squares"],
    ["potato", "late blight", "remove infected foliage"],
    ["maize", "fall armyworm", "check the whorl"],
    ["chickpea", "pod borer", "scout the flowers"],
    ["groundnut", "leaf spot", "rotate with cereals"],
    ["cabbage", "diamondback moth", "cover young plants"],
    ["chili", "thrips", "leaf curl", "control whiteflies"],
    ["banana", "sigatoka", "remove spotted leaves"],
    ["soybean", "rust", "scout lower leaves"],
    ["mustard", "aphids", "spray the flower heads"],
    ["onion", "thrips", "purple blotch", "widen the spacing"],
    ["okra", "whitefly", "yellow vein mosaic", "remove infected plants"],
    ["mango", "hoppers", "powdery mildew", "prune for airflow"],
    ["coffee", "berry borer", "pick fallen berries"],
    ["brinjal", "shoot borer", "clip wilted shoots"],
    ["sugarcane", "early shoot borer", "destroy dead hearts"],
    ["peas", "powdery mildew", "widen the spacing"],
]

ROTATION_SEQUENCES: list[list[str]] = [
    ["rice", "wheat", "moong"],
    ["cotton", "chickpea", "fallow"],
    ["maize", "mustard", "moong"],
    ["soybean", "wheat", "moong"],
    ["groundnut", "wheat", "fallow"],
    ["potato", "maize", "wheat"],
]

TEMPLATES: list[list[str]] = PEST_SEQUENCES + ROTATION_SEQUENCES

_BUNDLE: dict[tuple, dict[str, Any]] = {}


def build_sequences(repeats: int = 24, seed: int = 0) -> list[list[str]]:
    """Repeat each field template, then shuffle. Copies share a pattern; the order of copies does not."""
    sequences = [list(template) for template in TEMPLATES for _ in range(repeats)]
    order = np.random.default_rng(seed).permutation(len(sequences))
    return [sequences[int(i)] for i in order]


def as_baskets(sequences: list[list[str]]) -> list[list[str]]:
    """One basket per sequence. Repeated items count once, matching the miner."""
    return [list(dict.fromkeys(seq)) for seq in sequences]


def split_sequences(
    sequences: list[list[str]], test_ratio: float = 0.2, seed: int = 1
) -> tuple[list[list[str]], list[list[str]]]:
    order = np.random.default_rng(seed).permutation(len(sequences))
    cut = max(1, int(round(len(sequences) * (1 - test_ratio))))
    train = [sequences[int(i)] for i in order[:cut]]
    test = [sequences[int(i)] for i in order[cut:]]
    return train, test


# --- Milestone 1: Define agricultural sequences as baskets.


def milestone_1(
    sequences: list[list[str]] | None = None, repeats: int = 24, seed: int = 0
) -> dict[str, Any]:
    """Collect crop-rotation and pest-observation sequences and view them as baskets."""
    chosen = [list(seq) for seq in sequences] if sequences is not None else build_sequences(repeats, seed)
    cleaned = [[item.strip().lower() for item in seq if item.strip()] for seq in chosen]
    cleaned = [seq for seq in cleaned if seq]
    baskets = as_baskets(cleaned)
    vocabulary = sorted({item for seq in cleaned for item in seq})
    return {
        "sequences": cleaned,
        "baskets": baskets,
        "vocabulary": vocabulary,
        "num_sequences": len(cleaned),
        "num_baskets": len(baskets),
        "vocab_size": len(vocabulary),
        "num_templates": len(TEMPLATES) if sequences is None else len(cleaned),
    }


# --- Milestone 2: Mine frequent co-occurring patterns with thilimarketbasket.


def _rule_frame(found: dict[frozenset, float], min_confidence: float) -> pd.DataFrame:
    rows = []
    for antecedent, consequent, support, confidence, lift in rules(found, min_confidence):
        rows.append(
            {
                "antecedent": " + ".join(sorted(antecedent)),
                "consequent": " + ".join(sorted(consequent)),
                "support": support,
                "confidence": confidence,
                "lift": lift,
                "size": len(antecedent) + len(consequent),
            }
        )
    frame = pd.DataFrame(rows, columns=["antecedent", "consequent", "support", "confidence", "lift", "size"])
    if len(frame):
        frame = frame.sort_values(["lift", "support"], ascending=False).reset_index(drop=True)
    return frame


def milestone_2(
    sequences: list[list[str]] | None = None,
    min_support: float = 0.03,
    min_confidence: float = 0.4,
    max_size: int = 3,
) -> dict[str, Any]:
    """Find frequent itemsets and association rules in the agricultural baskets."""
    data = milestone_1(sequences)
    found = frequent_itemsets(data["baskets"], min_support, max_size=max_size)
    frame = _rule_frame(found, min_confidence)
    patterns = [
        {"items": sorted(items), "support": support, "size": len(items)}
        for items, support in sorted(found.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
        if len(items) >= 2
    ]
    return {
        "baskets": data["baskets"],
        "vocabulary": data["vocabulary"],
        "itemsets": found,
        "num_itemsets": len(found),
        "patterns": patterns,
        "rules": frame,
        "min_support": min_support,
        "min_confidence": min_confidence,
    }


# --- Milestone 3: Continued pretraining of a small next-token model.


def _pairs(sequences: list[list[str]], stoi: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev2: list[int] = []
    prev1: list[int] = []
    target: list[int] = []
    for seq in sequences:
        ids = [stoi[token] for token in seq]
        for index in range(1, len(ids)):
            prev1.append(ids[index - 1])
            prev2.append(ids[index - 2] if index >= 2 else -1)
            target.append(ids[index])
    return np.asarray(prev1, dtype=int), np.asarray(prev2, dtype=int), np.asarray(target, dtype=int)


class SequenceLM:
    """A tiny causal model: the next item depends on the previous two.

    Training is the same next-token objective as continued pretraining. There is no label besides
    the following item in the sequence.
    """

    def __init__(self, vocab: list[str], dim: int = 32, seed: int = 0) -> None:
        self.vocab = list(vocab)
        self.stoi = {token: index for index, token in enumerate(self.vocab)}
        width = len(self.vocab)
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.05, size=(width, dim))
        self.W = rng.normal(0, 0.05, size=(2 * dim, width))
        self.b = np.zeros(width)

    def _hidden(self, prev1: np.ndarray, prev2: np.ndarray) -> np.ndarray:
        latest = self.E[prev1]
        earlier_vec = np.zeros_like(latest)
        earlier = prev2 >= 0
        if np.any(earlier):
            earlier_vec[earlier] = self.E[prev2[earlier]]
        return np.concatenate([latest, earlier_vec], axis=-1)

    def loss_and_grad(
        self, prev1: np.ndarray, prev2: np.ndarray, target: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        hidden = self._hidden(prev1, prev2)
        logits = hidden @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        rows = len(target)
        picked = probs[np.arange(rows), target]
        loss = float(-np.log(picked + 1e-12).mean())
        grad_logits = probs.copy()
        grad_logits[np.arange(rows), target] -= 1.0
        grad_logits /= rows
        grad_w = hidden.T @ grad_logits
        grad_b = grad_logits.sum(axis=0)
        grad_h = grad_logits @ self.W.T
        half = self.E.shape[1]
        grad_e = np.zeros_like(self.E)
        np.add.at(grad_e, prev1, grad_h[:, :half])
        earlier = prev2 >= 0
        if np.any(earlier):
            np.add.at(grad_e, prev2[earlier], grad_h[earlier, half:])
        return loss, grad_e, grad_w, grad_b

    def fit(
        self, prev1: np.ndarray, prev2: np.ndarray, target: np.ndarray, epochs: int, lr: float
    ) -> list[float]:
        """Adam on the next-token loss. A few epochs is enough for this vocabulary."""
        losses = []
        moment_e = np.zeros_like(self.E)
        moment_w = np.zeros_like(self.W)
        moment_b = np.zeros_like(self.b)
        velocity_e = np.zeros_like(self.E)
        velocity_w = np.zeros_like(self.W)
        velocity_b = np.zeros_like(self.b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for step in range(1, epochs + 1):
            loss, grad_e, grad_w, grad_b = self.loss_and_grad(prev1, prev2, target)
            moment_e = beta1 * moment_e + (1 - beta1) * grad_e
            moment_w = beta1 * moment_w + (1 - beta1) * grad_w
            moment_b = beta1 * moment_b + (1 - beta1) * grad_b
            velocity_e = beta2 * velocity_e + (1 - beta2) * grad_e**2
            velocity_w = beta2 * velocity_w + (1 - beta2) * grad_w**2
            velocity_b = beta2 * velocity_b + (1 - beta2) * grad_b**2
            correct1 = 1 - beta1**step
            correct2 = 1 - beta2**step
            self.E -= lr * (moment_e / correct1) / (np.sqrt(velocity_e / correct2) + eps)
            self.W -= lr * (moment_w / correct1) / (np.sqrt(velocity_w / correct2) + eps)
            self.b -= lr * (moment_b / correct1) / (np.sqrt(velocity_b / correct2) + eps)
            losses.append(loss)
        return losses

    def probabilities(self, context: list[str]) -> np.ndarray:
        ids = [self.stoi[token] for token in context if token in self.stoi]
        if not ids:
            logits = self.b.copy()
        else:
            prev1 = np.array([ids[-1]])
            prev2 = np.array([ids[-2] if len(ids) > 1 else -1])
            logits = (self._hidden(prev1, prev2) @ self.W + self.b)[0]
        logits = logits - logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def rank(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        probs = self.probabilities(context)
        order = np.argsort(-probs, kind="stable")
        take = min(k, len(order))
        return [(self.vocab[int(index)], float(probs[int(index)])) for index in order[:take]]


def milestone_3(
    repeats: int = 24,
    seed: int = 0,
    pretrain_epochs: int = 6,
    epochs: int = 80,
    lr: float = 0.03,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Pretrain on shuffled next-items, then continue training on the real crop sequences."""
    key = (repeats, seed, pretrain_epochs, epochs, lr)
    if use_cache and key in _BUNDLE:
        return _BUNDLE[key]

    data = milestone_1(repeats=repeats, seed=seed)
    train, test = split_sequences(data["sequences"], seed=seed + 1)
    model = SequenceLM(data["vocabulary"], seed=seed)
    prev1, prev2, target = _pairs(train, model.stoi)
    rng = np.random.default_rng(seed + 7)
    noise = rng.integers(0, len(model.vocab), size=len(target))
    pretrain_loss = model.fit(prev1, prev2, noise, epochs=pretrain_epochs, lr=lr)
    domain_before = model.loss_and_grad(prev1, prev2, target)[0]
    domain_loss = model.fit(prev1, prev2, target, epochs=epochs, lr=lr)
    bundle = {
        "model": model,
        "vocabulary": data["vocabulary"],
        "train_sequences": train,
        "test_sequences": test,
        "pretrain_loss": pretrain_loss,
        "domain_before": domain_before,
        "domain_loss": domain_loss,
        "initial_domain_loss": domain_loss[0],
        "final_loss": domain_loss[-1],
        "epochs": epochs,
    }
    if use_cache:
        _BUNDLE[key] = bundle
    return bundle


# --- Milestone 4: Evaluate next-item predictions on unseen sequences.


def _cases(sequences: list[list[str]]) -> list[tuple[list[str], str]]:
    cases = []
    for seq in sequences:
        for index in range(1, len(seq)):
            cases.append((seq[:index], seq[index]))
    return cases


def _ranks(model: SequenceLM, sequences: list[list[str]]) -> list[int]:
    ranks = []
    for context, target in _cases(sequences):
        names = [name for name, _ in model.rank(context, k=len(model.vocab))]
        ranks.append(names.index(target) + 1 if target in names else len(model.vocab))
    return ranks


def _baseline_order(train_sequences: list[list[str]], vocab: list[str]) -> list[str]:
    counts = {token: 0 for token in vocab}
    for seq in train_sequences:
        for token in seq[1:]:
            counts[token] = counts.get(token, 0) + 1
    return sorted(vocab, key=lambda token: (-counts[token], token))


def milestone_4(bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score top-1, top-3, and mean reciprocal rank against a popularity baseline."""
    if bundle is None:
        bundle = milestone_3()
    model: SequenceLM = bundle["model"]
    ranks = _ranks(model, bundle["test_sequences"])
    rank_array = np.asarray(ranks, dtype=float)
    order = _baseline_order(bundle["train_sequences"], model.vocab)
    baseline_ranks = []
    for _context, target in _cases(bundle["test_sequences"]):
        baseline_ranks.append(order.index(target) + 1 if target in order else len(order))
    baseline = np.asarray(baseline_ranks, dtype=float)
    metrics = {
        "n_cases": int(len(ranks)),
        "top1_accuracy": float(np.mean(rank_array <= 1)),
        "top3_accuracy": float(np.mean(rank_array <= 3)),
        "mrr": float(np.mean(1.0 / rank_array)),
        "median_rank": float(np.median(rank_array)),
        "baseline_top1": float(np.mean(baseline <= 1)),
        "baseline_mrr": float(np.mean(1.0 / baseline)),
        "final_loss": bundle["final_loss"],
    }
    metrics["beats_baseline"] = metrics["mrr"] > metrics["baseline_mrr"]
    metrics["model"] = model
    metrics["bundle"] = bundle
    return metrics


# --- Milestone 5: CLI payload.


def milestone_5(
    context: list[str] | None = None, k: int = 5, bundle: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Rank the likely next observations for a sequence the user has already seen."""
    if bundle is None:
        bundle = milestone_3()
    model: SequenceLM = bundle["model"]
    tokens = [part.strip().lower() for part in (context or ["rice", "brown planthopper"]) if part.strip()]
    known = [token for token in tokens if token in model.stoi]
    unknown = [token for token in tokens if token not in model.stoi]
    ranked = model.rank(known, k=k)
    lines = [f"Observations: {', '.join(tokens) if tokens else '(none)'}"]
    if unknown:
        lines.append("Not in the field vocabulary: " + ", ".join(unknown))
    lines.append("Likely next:")
    for index, (name, prob) in enumerate(ranked, start=1):
        lines.append(f"  {index}. {name:<24} {prob:6.1%}")
    return {
        "context": tokens,
        "known": known,
        "unknown": unknown,
        "predictions": ranked,
        "report": "\n".join(lines),
        "top": ranked[0][0] if ranked else "",
    }


def pattern_report(min_support: float = 0.03, limit: int = 12) -> str:
    """Text bars for the strongest co-occurring pairs and triples."""
    mined = milestone_2(min_support=min_support)
    lines = ["Frequent co-occurrences:"]
    strongest = mined["patterns"][:limit]
    if not strongest:
        return "Frequent co-occurrences:\n  (none at this support)"
    peak = max(row["support"] for row in strongest)
    for row in strongest:
        width = max(1, round(24 * row["support"] / peak))
        label = " + ".join(row["items"])
        lines.append(f"  {label:<48} {'#' * width}  {row['support']:.1%}")
    return "\n".join(lines)
