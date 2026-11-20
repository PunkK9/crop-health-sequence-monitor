"""Tests for the five crop-health milestones."""

import pytest

from app.cli import main
from app.core import milestone_1, milestone_2, milestone_3, milestone_4, milestone_5


@pytest.fixture(scope="module")
def bundle():
    return milestone_3(repeats=24, epochs=80, seed=0)


def test_milestone_1():
    result = milestone_1()
    assert result["num_baskets"] == result["num_sequences"]
    assert result["num_sequences"] > 100
    assert "rice" in result["vocabulary"]
    assert "brown planthopper" in result["vocabulary"]
    assert all(isinstance(basket, list) and basket for basket in result["baskets"])

    custom = milestone_1([["  Tomato ", "Whitefly"], ["wheat", "stripe rust"]])
    assert custom["num_sequences"] == 2
    assert custom["baskets"][0] == ["tomato", "whitefly"]
    assert "tomato" in custom["vocabulary"]


def test_milestone_2():
    mined = milestone_2(min_support=0.03)
    assert mined["num_itemsets"] > 10
    pair = frozenset({"rice", "brown planthopper"})
    assert pair in mined["itemsets"]
    assert mined["itemsets"][pair] > 0.03
    assert len(mined["patterns"]) > 0
    assert {"antecedent", "consequent", "support", "confidence", "lift"} <= set(mined["rules"].columns)
    rice_rules = mined["rules"][mined["rules"]["antecedent"] == "brown planthopper"]
    assert not rice_rules.empty
    assert (rice_rules["confidence"] >= 0.4).any()


def test_milestone_3(bundle):
    assert bundle["domain_before"] > bundle["final_loss"]
    assert bundle["domain_loss"][-1] < bundle["domain_loss"][0]
    assert len(bundle["pretrain_loss"]) == 6
    assert bundle["test_sequences"]
    probs = bundle["model"].probabilities(["rice", "brown planthopper"])
    assert probs.shape == (len(bundle["vocabulary"]),)
    assert probs.sum() == pytest.approx(1.0)


def test_milestone_4(bundle):
    metrics = milestone_4(bundle)
    assert metrics["n_cases"] > 50
    assert metrics["top1_accuracy"] > 0.6
    assert metrics["top3_accuracy"] > 0.8
    assert metrics["beats_baseline"]
    assert metrics["mrr"] > metrics["baseline_mrr"]


def test_milestone_5(bundle, capsys):
    result = milestone_5(["Rice", "brown planthopper"], k=3, bundle=bundle)
    assert result["top"] == "sheath blight"
    assert "Likely next:" in result["report"]
    assert len(result["predictions"]) == 3

    main(["next", "rice", "brown planthopper", "--k", "3"])
    output = capsys.readouterr().out
    assert "sheath blight" in output
    assert "Likely next:" in output

    main(["patterns", "--limit", "5"])
    patterns = capsys.readouterr().out
    assert "Frequent co-occurrences:" in patterns
    assert "wheat" in patterns

    main(["evaluate"])
    stats = capsys.readouterr().out
    assert "top-1 accuracy" in stats
    assert "MRR" in stats
