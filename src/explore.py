import pandas as pd
import random
import os
from PIL import Image

df = pd.read_csv('data/styles.csv', on_bad_lines='skip')
print("Shape: ", df.shape) #22 bad rows


# print(df[0:5])

sample=df.sample(n=5000)
ids=[sample.iloc[random.randint(0, len(sample))]['id'] for i in range(10)]
# for id in ids:
    # print(os.path.exists(f"data/images/{id}.jpg"))


sample_5=[sample.iloc[random.randint(0, len(sample))] for i in range(5)]
# for r in sample_5:
#     img=Image.open(f"data/images/{r['id']}.jpg")
#     img.show()
#     print(r['gender'], r['articleType'], r['baseColour'], r['productDisplayName'])

print("Missing values per column: ", df.isnull().sum())