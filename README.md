# fashion-similarity-search
PyTorch implementation of a fashion visual similarity search engine using Fashion Product Images
-> starting with FPI (small)

```bash
pip install torch torchvision faiss-cpu numpy pillow matplotlib pandas kaggle jupyter scikit-learn

mkdir data
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/

cd data && unzip fashion-product-images-small.zip
```

## Before (FULL-backbone) finetuning:

| Category | P@1 | P@3 | P@5 | P@10 | ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| same_article_type | 0.900 | 0.967 | 0.940 | 0.910 | 6.33ms |
| same_color | 0.500 | 0.500 | 0.480 | 0.450 | 5.92ms |
| same_gender | 0.800 | 0.767 | 0.780 | 0.810 | 6.09ms |
| footwear | 1.000 | 1.000 | 1.000 | 1.000 | 5.84ms |
| outerwear | 0.600 | 0.600 | 0.480 | 0.430 | 5.77ms |
| tops | 1.000 | 1.000 | 1.000 | 1.000 | 5.89ms |
| bottom_wear | 1.000 | 1.000 | 1.000 | 1.000 | 5.94ms |
| accessories | 1.000 | 1.000 | 1.000 | 1.000 | 7.17ms |
| bags | 1.000 | 1.000 | 1.000 | 0.960 | 5.81ms |
| jewelry | 1.000 | 1.000 | 1.000 | 1.000 | 5.84ms |
| innerwear | 1.000 | 0.967 | 0.940 | 0.940 | 5.79ms |
| casual_wear | 0.600 | 0.800 | 0.820 | 0.880 | 5.96ms |
| dresses | 0.900 | 0.733 | 0.700 | 0.670 | 5.81ms |
| rare_class | 1.000 | 0.833 | 0.820 | 0.830 | 5.78ms |
| **MEAN** | **0.879** | **0.869** | **0.854** | **0.849** | **6.00ms** |


- mean P@5 = 0.854
- my P@5 = 1.000 cases: footwear, tops, bottom_wear, accessories, bags, jewelry (all visually distinctive)
- what's lagging:
    - same color (P@5 = .480): color-based retrieval is a known weakness of classification-pretrained embeddings
    - outerwear (P@5 = .480): even though jackets/sweaters/sweatshirts are visually similar to each other, they are also visually similar to shirts/tops
    - dresses (P@5 = .700): in a qualitiative check, I noticed there often is confusion between dresses and tops
    - same gender (P@5 = .780): decent, but perhaps gender isn't as strong of a visual signal
- latency: ~5.996ms per query


### With IndexIVFFlat

| Category | P@1 | P@3 | P@5 | P@10 | ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| same_article_type | 0.900 | 0.967 | 0.940 | 0.910 | 3.04ms |
| same_color | 0.500 | 0.500 | 0.480 | 0.450 | 1.45ms |
| same_gender | 0.800 | 0.767 | 0.780 | 0.810 | 0.77ms |
| footwear | 1.000 | 1.000 | 1.000 | 1.000 | 0.72ms |
| outerwear | 0.600 | 0.600 | 0.480 | 0.430 | 0.81ms |
| tops | 1.000 | 1.000 | 1.000 | 1.000 | 0.80ms |
| bottom_wear | 1.000 | 1.000 | 1.000 | 1.000 | 0.59ms |
| accessories | 1.000 | 1.000 | 1.000 | 1.000 | 0.74ms |
| bags | 1.000 | 1.000 | 1.000 | 0.960 | 0.59ms |
| jewelry | 1.000 | 1.000 | 1.000 | 1.000 | 0.47ms |
| innerwear | 1.000 | 0.967 | 0.940 | 0.940 | 0.77ms |
| casual_wear | 0.600 | 0.800 | 0.820 | 0.890 | 0.79ms |
| dresses | 0.900 | 0.733 | 0.700 | 0.670 | 0.79ms |
| rare_class | 1.000 | 0.833 | 0.820 | 0.830 | 0.58ms |
| **MEAN** | **0.879** | **0.869** | **0.854** | **0.849** | **.92ms** |

| Method | Mean P@5 | Avg Latency |
| --- | ---: | ---: |
| IndexFlatL2 (exact) | 0.854 | ~6ms |
| IndexIVFFlat (approximate) | 0.854 | ~0.8ms |
| **Speedup** | **0% precision loss** | **~7.5× faster** |


## Results after full finetuning
- dresses (+0.160): This category in baseline was often confused with Tops. Full fine-tuning learned to distinguish these
- same_gender (+0.100): The model learned gender as a visual signal, which a frozen backbone couldn't do
- casual_wear, innerwear, rare_class: small gains

- footwear (-0.080) and accessories (-0.120): these were at 1.00 in P@1 but dropped in P@5, meaning the fine-tuned model is slightly more "opinionated"
    - finetuned model retrieves very close matches first but the 4th/5th results are less diverse. Reflective of a known fine-tuning tradeoff: better precision at top-1, slightly narrower retrieval overall

Frozen backbone fine-tuning improved classification accuracy (82% -> 88.95%) but didn't move embeddings. Full backbone fine-tuning moved both: embedding clustering improved (same-class L2: 8.06 -> 12.92, but ratio improved), and the weakest retrieval category (dresses) saw the largest gain (+16%).

| Category | Base P@5 | FT P@5 | Delta |
|----------|---------:|--------:|------:|
| same_article_type | 0.940 | 0.920 | -0.020 |
| same_color | 0.480 | 0.460 | -0.020 |
| same_gender | 0.780 | 0.880 | +0.100 |
| footwear | 1.000 | 0.920 | -0.080 |
| outerwear | 0.480 | 0.460 | -0.020 |
| tops | 1.000 | 1.000 | +0.000 |
| bottom_wear | 1.000 | 1.000 | +0.000 |
| accessories | 1.000 | 0.880 | -0.120 |
| bags | 1.000 | 0.980 | -0.020 |
| jewelry | 1.000 | 1.000 | +0.000 |
| innerwear | 0.940 | 0.960 | +0.020 |
| casual_wear | 0.820 | 0.840 | +0.020 |
| dresses | 0.700 | 0.860 | +0.160 |
| rare_class | 0.820 | 0.840 | +0.020 |
| **Mean** | **0.854** | **0.857** | **+0.003** |


| Model | Avg Latency (Flat Index) |
|-------|------------:|
| Baseline | ~6.00 ms |
| Fine-tuned | ~6.12 ms |

| Index | Mean P@5 | Avg Latency |
|-------|---------:|------------:|
| IndexFlatL2 (exact) | 0.854 | ~6.00 ms |
| IndexIVFFlat (approximate) | 0.854 | ~0.92 ms |
| **Speedup** | **0% precision loss** | **~6.5× faster** |