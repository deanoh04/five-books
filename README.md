# Five Books Is Enough

**[Live demo →](https://five-books.vercel.app/)**

Name five books someone has rated, even with a few stars misremembered, and this model picks them out of **53,424 real Goodreads readers** with **97.4% accuracy**. Exact star matching on the same test gets 15.0%.

The demo runs entirely in the browser. Draw a random real reader, show the model 5 of their ratings, and watch it find them. Then try to hide a reader yourself.

## Results

All numbers are on held-out readers the learned model never trained on (3,000 per headline number). Attacker knows k books from the target's shelf; "noisy" means each star is off by one with 30% probability.

| Scenario | Exact match | Rarity scoring | Learned model |
|---|---|---|---|
| 5 books, noisy | 15.0% | 92.7% | **97.4%** (±0.3) |
| 5 books, exact | 99.4% | 99.4% | **99.6%** |
| 3 books, noisy | 26.8% | 58.7% | **68.7%** |
| 3 books, exact | 80.4% | 80.4% | **85.2%** |
| Only the target's 5 most popular books, noisy | 5.6% | 12.2% | **15.5%** |
| Only the target's 8 most popular books, noisy | 4.0% | 43.4% | **49.5%** |

**Defense experiment.** Randomizing a share of ratings before release (randomized response). Recommender = ALS matrix factorization, error measured on held-out true ratings. Predicting each book's average scores 0.955 RMSE.

| Ratings randomized | Re-identified (learned) | Re-identified (rarity) | Recommender RMSE |
|---|---|---|---|
| 0% | 99.0% | 96.6% | 0.861 |
| 10% | 93.4% | 79.9% | 0.920 |
| 25% | 84.8% | 59.8% | 1.023 |
| 50% | 71.0% | 38.4% | 1.205 |
| 75% | 47.6% | 15.0% | 1.373 |

Defense rows use 500 readers each drawn from all readers, so the 0% row reads a little higher than the held-out headline number.

Naive noise makes recommendations worse than a trivial baseline (at 25%) while the learned attacker still re-identifies 85% of readers.

**Calibration.** Confidence comes from a softmax with a fitted temperature (1.55). Expected calibration error on held-out readers: 1.3 points.

## How it works

- **Threat model** (as in Narayanan & Shmatikov, 2008, on the Netflix Prize data): the attacker has the full "anonymized" ratings table and knows a few (book, approximate stars) facts about one person.
- **Features** for every reader who shares a clue book: rarity-weighted exact / off-by-one / big-mismatch matches (weight = 1 / log(readers of that book)), raw match counts, and log shelf size.
- **Model**: conditional logit (listwise softmax over candidate readers) trained with L-BFGS on 3,000 simulated attacks, using the 400 hardest look-alike readers per attack as negatives. Train and test readers are disjoint.
- **In-browser engine**: the 6M ratings are packed into 128 book shards and 107 reader shards (uint16 ids, uint8 stars). A search fetches only the shards for its clue books and scores all 53,424 readers in about 5 ms. The TypeScript engine was checked against the Python model on 300 random queries: identical rankings on all 300.

## Deploy on Vercel

1. Push this folder to a GitHub repo.
2. In Vercel, **Add New → Project**, import the repo, and click **Deploy**. The root `vercel.json` already sets the install/build commands and output folder, so no settings are needed.

Run locally: `cd web && npm install && npm run dev`.

## Reproduce the model

```bash
cd model
pip install -r requirements.txt
./get_data.sh            # downloads goodbooks-10k
python train_eval.py     # trains the ranker, writes results.json
python calibrate.py      # fits the confidence temperature
python defense.py        # privacy defense experiment
python headline.py       # 3,000-reader headline numbers
python export_web.py     # writes web/public/data
```

## Data

[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) by Zygmunt Zając, CC BY-SA 4.0. Readers are anonymous numeric ids in the source data. Book covers load from Goodreads' image CDN when available; otherwise a generated cover is shown.
