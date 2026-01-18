# Vision Transformers

Referencess:
- Video tutorial: https://www.youtube.com/watch?v=vAmKB7iPkWw

### From CLIP to SigLIP:
- Softmax has an exponent expression, which can cause overflow on hardware. Adding a log fn in the exponent makes it less likely to overflow.
- Normalization of each logit depends on the entire row or entire column, making it hard to batch/parallelize on hardware.
- SigLIP uses sigmoid instead of cross entropy so that each logit is a binary classification, independent on other logits. Hence, making it easier to optimize on hardware.
- The encoder can be used alongside an LLM to merge text and image features to produce text. AKA Vision Language Model.

### Normalization:
- Without normalization, covariate shift happens: Big change in input -> Big change in outputs of layers -> Big change in loss -> Big change in gradients -> Big change in the weights of the network -> Network learns slowly.
- The model spends most of the epoch trying to learn the change in distribution rather than the features.
- Solution? Normalization so the inputs will always follow a distribution (Gaussian).
- BatchNorm: calculate mean and variance along the batch dimension, across inputs. Problem is we require more inputs per batch.
- LayerNorm: calculate mean and variance along each layer, which means it is input independent.