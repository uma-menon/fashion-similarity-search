# fashion-similarity-search
PyTorch implementation of a fashion visual similarity search engine using Fashion Product Images
-> starting with FPI (small)

```bash
pip install torch torchvision faiss-cpu numpy pillow matplotlib pandas kaggle jupyter scikit-learn

mkdir data
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/

cd data && unzip fashion-product-images-small.zip
```

## Post-finetuning, re-index and query:
Queried across the following categories: Dresses, Heels, Jeans, Sunglasses, Tshirts

For each query, note: how many of the top 5 results are the correct class? Are they visually similar beyond just category (same color, similar silhouette)? Does the finetuned version do better, worse, or the same?

### DRESSES: queried on a straight, black, square-neck midi dress with a high neckline
#### Baseline: 
- 4 out of 5 results were dresses (outlier was a tshirt)
- 3 results were black, 2 navy
- all 4 dress results were roughly the same lenght
- varied necklines

#### Finetuned:
- 2 out of 5 results were dresses
- 2 results were black, 2 were green (1 of which had a black bottom), 1 was pink
- 4 out of 5 results roughly the same length
