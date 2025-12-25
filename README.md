# DNN_PROJECT

# Graph-Based Entity Reasoning for Visual Story Continuation

Purpose
- Improve multimodal story continuation by modeling entity relations across time and fusing graph signals into sequence prediction.

Method overview
- Image context: ResNet18 embeddings (normalized) for each context frame.
- Text context: TF-IDF vectors projected to 512-dim embeddings.
- Sequence model: GRU over fused image+text embeddings to predict next-step image/text embeddings.
- Graph reasoning: entity extraction from `<gdo ...>` tags, graph construction (co-occurrence + temporal proximity), gated message passing, and graph pooling.
- Fusion variants: full graph fusion and a text-only graph fusion option.

Repository layout
- `src/dataloader.py`: windowed dataset, TF-IDF collators, graph construction and padding.
- `src/encoders_image.py`: ResNet18 embedder.
- `src/encoders_text.py`: TF-IDF vectorizer utilities and text cleaning.
- `src/model_baseline.py`: ResNet+TF-IDF GRU baseline.
- `src/graph_module.py`: simple gated graph reasoner.
- `src/model_graph.py`: graph-fused predictors.
- `src/train.py`: training loop for baseline/graph modes.
- `src/eval.py`: validation metrics helpers.
- `src/download_dataset.py`: Hugging Face StoryReasoning download/inspection.
- `src/prepare_manifest.py`: index preparation for K-step windows.

Quickstart
```bash
cd "Mani - Project 1"

# Baseline (ResNet + TF-IDF)
python src/train.py --mode baseline

# Graph-fused
python src/train.py --mode graph

# Graph fused into text head only
python src/train.py --mode graph_textonly
```

Notes
- The graph pipeline expects entity tags in the text like `<gdo charX>NAME</gdo>` to extract entities reliably.

