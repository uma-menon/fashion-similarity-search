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

### DRESSES: queried on a straight, black, scoop-neck midi dress (69987)
#### Baseline: 
- 2 out of 5 results were dresses
- both dress results were black, 2 tops were dark grey, 1 red top
- both dress results were roughly the same lengths
- 2 scoop-necks and 3 v-necks
- 5 out of 5 women's products

#### Finetuned:
- 2 out of 5 results were dresses
- all 5 results were black (1 potentially dark grey)
- 4 out of 5 results roughly the same length
- 1 scoop neck, 2 v-necks, 1 collared
- only 3 out of 5 were women's products
- 1 pair of shorts

### HEELS: queried on a silver, strappy, low wedge
#### Baseline: 
- all results were heels???
- 3 out of 5 results were silver, 1 white, 1 brown
- all results were strappy

#### Finetuned:
- 