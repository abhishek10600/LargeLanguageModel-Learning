## What are embeddings ?

As neural networks cannot process text directly, so textual data needs to be converted into numerical value
The concept of converting text data into a vector format is called embeddings.

## What are vectors ?

A vector in deep learning is an ordered list of numbers that represents data, features, weights, or hidden states in a format that neural networks can process.

## What is encoding ?

Encoding in machine learning is the process of converting non-numerical data, like categories or text, into numbers so algorithms can process them

## What is BytePair encoding ?

Byte pair encoding is an encoding algorithm for tokenizer that is popularly used in GPT-1, GPT-2, GPT-3 and GPT-4, it is also used by meta ai in their llama 3 model.

Byte Pair Encoding (BPE) is a data compression and subword tokenization algorithm that iteratively merges the most frequent pairs of consecutive bytes or characters in a text.

To implement BPE we will use tiktoken library.

BPE tokenizers break down unknown words into subwords and individual characters.
