# Simple MNIST NN (NumPy)

Educational repository implementing a two-layer neural network trained from scratch using NumPy on the MNIST digit recognition dataset (CSV format). The goal is to demonstrate the fundamental steps of an ML pipeline without TensorFlow or PyTorch.

## Repository contents

- `nn.ipynb`: Main notebook showing data loading, model definition, training and evaluation.
- `train.csv`, `test.csv`: Dataset files (MNIST) used for training and testing.

## Requirements

- Python 3.8+
- Python packages: `numpy`, `pandas`, `matplotlib`

Quick install:

```bash
python -m pip install --upgrade pip
pip install numpy pandas matplotlib jupyter
```

## Usage

1. Open the notebook: [nn.ipynb](nn.ipynb)
2. Make sure `train.csv` and `test.csv` are in the same folder as the notebook.
3. Run the notebook cells top to bottom.

The notebook implements:

- Data loading and preparation (normalization, train/dev split).
- Model functions: `init_params`, `forward_prop`, `softmax`, `ReLU`, `backward_prop`, `update_params`.
- Training loop: `gradient_descent`.
- Utilities: `get_predictions`, `get_accuracy`, `make_predictions`, `test_prediction`.

## Expected results

With the notebook's current settings (random initialization, learning rate and iterations), you can typically expect roughly ~85% training accuracy and ~84% accuracy on the development set.

## Suggested improvements

- Improve numerical stability of `softmax` (subtract max logits, use axis-aware sums).
- Use better weight initialization (He/Xavier) for `W1` and `W2`.
- Try alternative optimizers and hyperparameters (mini-batch SGD, Adam).
- Add regularization (dropout, L2) or batch normalization.
- Vectorize and optimize operations to speed up training.