---
base_model: unknown
library_name: model2vec
license: mit
model_name: potion-base-8M-model2vec-pipeline
tags:
- embeddings
- static-embeddings
- sentence-transformers
---

# potion-base-8M-model2vec-pipeline Model Card

This [Model2Vec](https://github.com/MinishLab/model2vec) model is a fine-tuned version of the [unknown](https://huggingface.co/unknown) Model2Vec model. It also includes a classifier head on top.

## Installation

Install model2vec using pip:
```
pip install model2vec[inference]
```

## Usage
Load this model using the `from_pretrained` method:
```python
from model2vec.inference import StaticModelPipeline

# Load a pretrained Model2Vec model
model = StaticModelPipeline.from_pretrained("potion-base-8M-model2vec-pipeline")

# Predict labels
predicted = model.predict(["Example sentence"])
```

## Additional Resources

- [Model2Vec Repo](https://github.com/MinishLab/model2vec)
- [Model2Vec Base Models](https://huggingface.co/collections/minishlab/model2vec-base-models-66fd9dd9b7c3b3c0f25ca90e)
- [Model2Vec Results](https://github.com/MinishLab/model2vec/tree/main/results)
- [Model2Vec Docs](https://minish.ai/packages/model2vec/introduction)

## Library Authors

Model2Vec was developed by the [Minish Lab](https://github.com/MinishLab) team consisting of [Stephan Tulkens](https://github.com/stephantul) and [Thomas van Dongen](https://github.com/Pringled).

## Citation

Please cite the [Model2Vec repository](https://github.com/MinishLab/model2vec) if you use this model in your work.
```
@software{minishlab2024model2vec,
  author       = {Stephan Tulkens and {van Dongen}, Thomas},
  title        = {Model2Vec: Fast State-of-the-Art Static Embeddings},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17270888},
  url          = {https://github.com/MinishLab/model2vec},
  license      = {MIT}
}
```