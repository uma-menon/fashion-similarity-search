import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import gradio as gr
from torchvision.models import resnet50
from PIL import Image

from dataset import clean, label_encode, transformations

faiss.omp_set_num_threads(1)

df = clean()
class_names, _ = label_encode(df)
df = df.set_index(df["id"].astype(str))

id_list = np.load("data/id_list_full_finetuned.npy", allow_pickle=False).tolist()
index = faiss.read_index("data/index_finetuned_full.index")

model = resnet50(weights=None)
model.fc = nn.Linear(2048, len(class_names))
model.load_state_dict(torch.load("data/best_model_fulltune.pt", map_location="cpu"))
model.fc = nn.Identity()
model.eval()

SAMPLES = [f"data/images/{i}.jpg" for i in (39716, 54118, 39386, 16957, 53759)]


def get_top_5(image_path):
    if image_path is None:
        return []

    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        embedding = model(transformations(img).unsqueeze(0))
    embedding = F.normalize(embedding, dim=1)

    query_id = Path(image_path).stem  # excludes the query itself when it's a dataset sample
    _, indices = index.search(embedding.numpy(), 6)

    results = []
    for i in indices[0]:
        img_id = str(id_list[i])
        if img_id == query_id:
            continue
        row = df.loc[img_id]
        caption = f"{row['productDisplayName']} | {row['baseColour']} | {row['articleType']}"
        results.append((f"data/images/{img_id}.jpg", caption))
    return results[:5]


if __name__ == "__main__":
    with gr.Blocks(title="Fashion Similarity Search Engine") as demo:
        gr.Markdown("## Fashion Similarity Search Engine")

        results = gr.Gallery(show_label=False, columns=5, render=False)

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("**Upload a clothing or accessory image**")
                image_input = gr.Image(type="filepath", show_label=False)
            with gr.Column(scale=1):
                gr.Markdown("**Or select a sample**")
                examples = gr.Examples(SAMPLES, inputs=image_input, outputs=results, fn=get_top_5, run_on_click=True)
                examples.dataset.show_label = False

        gr.Markdown("**Similar products:**")
        results.render()
        image_input.change(fn=get_top_5, inputs=image_input, outputs=results)

    demo.launch(theme=gr.themes.Default(primary_hue="rose"))
