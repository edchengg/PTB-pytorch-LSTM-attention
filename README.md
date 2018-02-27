# PTB Language Modelling task with RNNS(LSTM,GRU) and Attention Layer

This repository is used for a language modelling pareto competition at TTIC. 
I implemented an attention layer with the RNN model. 
TODO: (Lei Mao suggests another way to implement the attention layer by breaking into the LSTM class.)

## Software Requirements

This codebase requires Python 3, [PyTorch](http://pytorch.org/)

## Usage

```bash
python main.py --att --att_width 20        # Train a LSTM on PTB with attention layer and set the width of attenion to 20
python generate.py                      # Generate samples from the trained LSTM model.
```

## Acknowledge
This repository contains the code originally forked from the [Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model) that is modified to present attention layer into the model.
