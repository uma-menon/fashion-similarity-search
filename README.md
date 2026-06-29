# fashion-similarity-search
PyTorch implementation of a fashion visual similarity search engine using Fashion Product Images
-> starting with FPI (small)

```bash
pip install torch torchvision faiss-cpu numpy pillow matplotlib pandas kaggle jupyter

mkdir data
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/

cd data && unzip fashion-product-images-small.zip
```
