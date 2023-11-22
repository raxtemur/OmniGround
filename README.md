# OmniFusion

**OmniFusion** is an advanced multimodal AI model designed to extend the capabilities of traditional language processing systems by integrating additional data modalities such as images, and potentially audio, 3D and video content.

### ChangeLog
[22/11/2023] OmniFusion weights will soon be available!

### Architecture

<p align="left">
<img src="./content/architecture.png" width="100%">
</p>


OmniFusion open source version core is Mistral-7B. Initially focusing on images, we selected the CLIP-ViT-L as the visual encoder for its efficient information transfer capabilities. The most important component of OmniFusion is its adapter, a mechanism allowing the language model to interpret and incorporate information from different modalities. The adapter is a single-layer, four-headed transformer, which has shown superior performance compared to simpler linear layers or MLP structures.

This adapter takes embeddings from the visual encoder (excluding the CLS token) and maps them into textual embeddings compatible with the language model.

To further enhance the model's multimodal capabilities, we employ trainable special tokens to mark the beginning and end of visual data within the text sequence.


### Training Process consists of two stages

1. Pre-training the adapter on Image Captioning tasks (LAION, CC-4M).
2. Once the adapter has learned to map ViT's visual embeddings to the language model's textual space, we proceed to unfreeze Mistral for improved understanding of dialog formats and complex queries.

<p align="left">
<img src="./content/datasets.png" width="80%">
</p>

### Results

OmniFusion was benchmarked against the latest multimodal SOTA models. It excelled in generative metrics and classification benchmarks like VisualDialog.
<p align="left">
<img src="./content/radar.png" width="50%">
</p>

Model Performance on Visual Dialog Benchmark

| Model        | NDCG | MRR  | Recall@1 | Recall@5 | Recall@10 |
| ------------ | ---- | ---- | -------- | -------- | --------- |
| OmniFusion   | 25.91| 10.78| 4.74     | 13.80    | 20.53     |
| LLaVA-13B    | 24.74| 8.91 | 2.98     | 10.80    | 18.02     |

### Examples

<p align="left">
<img src="./content/examples.png" width="100%">
</p>

### Future Plans

We will soon release a public version of OmniFusion based on an open language model. Work is underway on a version that understands Russian, uses ImageBind encoders, and accepts more modalities (sound, 3D, video). Stay tuned for updates on GitHub!

### Authors

The FusionBrain scientific group from the AIRI Institute, in collaboration with scientists from Sber AI, led the model's development.

Main contributors:
+ Anton Razzhigaev: [Blog](https://t.me/abstractDL)
+ Elizaveta Goncharova
+ Matvey Mihkalchuk
+ Maxim Kurkin
+ Irina Abdullaeva
+ Denis Dimitrov [Blog](https://t.me/dendi_math_ai)
+ Andrey Kuznetsov [Blog](https://t.me/complete_ai)
