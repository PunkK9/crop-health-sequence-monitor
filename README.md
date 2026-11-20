[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/cxpvwcgsp7/badge.svg)](https://ask.thili.ai/b/cxpvwcgsp7)

# Crop Health Sequence Monitor

Farmers track crop health to predict co-occurring pests or the next crop in a rotation. Observations are stored as baskets, mined with `thilimarketbasket`, and used to continued-pretrain a small next-item language model. Ask with the crops and pests you have already seen, and the tool ranks what is likely to come next.

## What you are building

**Goal.** You will build a Crop Health Sequence Monitor that identifies common patterns in agricultural sequences and suggests next actions, helping farmers predict co-occurring pests or optimal crop sequences.

**Done when.** The project is complete when you have successfully applied frequent itemset mining and language model continued pretraining to agricultural sequences, and the model's next-item predictions are evaluated on new, unseen data.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Define agricultural sequences (e.g., crop rotations, pest observations) as 'baskets'.**  
      Hint: Structure your agricultural data into 'baskets' or sequences, similar to transaction data in `lfretail-lfmarketbasket`.
- [x] **2. Mine frequent co-occurring patterns using `thilimarketbasket`.**  
      Hint: Apply `thilimarketbasket` to find frequent itemsets representing common co-occurring patterns.
- [x] **3. Train a small LM on these sequences for next-item prediction.**  
      Hint: Use the language model continued pretraining technique to predict the next item in a sequence.
- [x] **4. Evaluate model's predictions on new, unseen agricultural sequences.**  
      Hint: Design metrics to evaluate the accuracy of your LM's next-item predictions on a held-out test set.
- [x] **5. Implement a CLI to query likely next items based on current observations.**  
      Hint: Build a command-line interface that takes a sequence as input and outputs the LM's predicted next items.

## Run it

You need Python 3.12 and git.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m app.cli hello
pytest
```

On macOS or Linux, activate with `source .venv/bin/activate` and copy the settings file with `cp .env.example .env`.

```bash
python -m app.cli next rice "brown planthopper"
python -m app.cli next wheat "stripe rust" --k 3
python -m app.cli patterns
python -m app.cli evaluate
```

Example:

```text
Observations: rice, brown planthopper
Likely next:
  1. sheath blight             85.0%
  2. drain the field            4.0%
```

## How it is built

- Sequences are ordered field notes: a crop, the pests that follow, and a next action, plus a few crop rotations.
- Each sequence is also a basket. `thilimarketbasket.mine.frequent_itemsets` and `rules` find the co-occurrences.
- The language model is continued-pretrained with the next-token loss: a short pass on shuffled targets, then training on the real sequences. The library's MLX trainer needs an Apple GPU, so this next-token trainer runs in NumPy on Windows and on GitHub Actions.
- Held-out copies of the sequences are scored with top-1, top-3, and mean reciprocal rank, against a popularity baseline that ignores the current sequence.

## Settings and secrets

Model settings for the unused hosted-chat helper live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.**

## Stretch ideas

- Implement a simple web interface for easier interaction beyond the CLI.
- Explore different language model architectures or fine-tuning parameters to improve prediction accuracy.
- Visualize the most frequent co-occurring patterns to provide deeper insights.

`patterns` prints the strongest itemsets as text bars. `next --k` and `evaluate` are the query and the held-out score.
