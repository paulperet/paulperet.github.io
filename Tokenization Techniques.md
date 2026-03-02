---
title: Tokenization techniques
layout: page
grand_parent: 
---

## Tokenization techniques

Before diving into the complex architecture of GPT-2, we need to understand the role of tokenization. We need to map each unique character or word into a single number to create our vocabulary. Our vocabulary consists of all mappings $(token, id)$. A token is a number (id) that corresponds to a unique string.

> Why don't we just use character-level tokens like the classical encoding format ASCII or UTF-8 ?

The motivation behind tokenization is to help the LLM learns meaningful representations of the words by pushing it to reason about them at a higher level. Indeed, humans rarely think about the characters composing the word when articulating a thought.

### Word-Level tokenization

This is the simplest form of tokenization. It maps entire words to an id.

<div style="display: flex; gap: 20px;justify-content: center;
align-items: center;">
  <img src="images/Word-Level-Tokenization.png" style="width: 80%; height: auto;">
</div>

### Byte-Pair Encoding

Originally, Byte-Pair Encoding is an algorithm used to compress text. The algorithm starts by taking all the unique characters in the corpus. Then it merges the most frequent pairs until a vocabulary size is reached. During inference, this merging process is repeated until no further merge can be done.

#### Example

I think that BPE is better explained through an example. For our example we will tokenize this single sentence :

```python
sentence = "The goal of this tutorial is to learn about different tokenization techniques."
```

First we start by splitting every word of our corpus. 

>The splitting rule for GPT-2 is special as it represents space as a special character, rendered as 'Ġ' and incorporate it to the beginning of the word that follows it. It also seperate punctuation from the words.

```python
corpus = {'The': 1, 'Ġgoal': 1, 'Ġof': 1, 'Ġthis': 1, 'Ġtutorial': 1, 'Ġis': 1, 'Ġto': 1, 'Ġlearn': 1, 'Ġabout': 1, 'Ġdifferent': 1, 'Ġtokenization': 1, 'Ġtechniques': 1 ,'.': 1}
```

We start with a vocabulary that comprises of all single characters. We will expand this vocabulary at each iteration of our algorithm.

```python
vocabulary = ['u', 'h', 's', 'i', 't', 'q', 'o', 'r', 'T', 'l', 'n', 'f', 'c', '.', 'z', 'e', 'b', 'Ġ', 'd', 'a', 'k', 'g']
```

If we apply our merging rules before running the algorithm, we just obtain a character level tokenization:

<div style="display: flex; gap: 20px;justify-content: center;
align-items: center;">
  <img src="images/BPE-Step0-Tokenization.png" style="width: 80%; height: auto;">
</div>

We split invidual words into single characters :

```python
{'The': ['T', 'h', 'e'],
 'Ġgoal': ['Ġ', 'g', 'o', 'a', 'l'],
 'Ġof': ['Ġ', 'o', 'f'],
 'Ġthis': ['Ġ', 't', 'h', 'i', 's'],
 'Ġtutorial': ['Ġ', 't', 'u', 't', 'o', 'r', 'i', 'a', 'l'],
 'Ġis': ['Ġ', 'i', 's'],
 'Ġto': ['Ġ', 't', 'o'],
 'Ġlearn': ['Ġ', 'l', 'e', 'a', 'r', 'n'],
 'Ġabout': ['Ġ', 'a', 'b', 'o', 'u', 't'],
 'Ġdifferent': ['Ġ', 'd', 'i', 'f', 'f', 'e', 'r', 'e', 'n', 't'],
 'Ġtokenization': ['Ġ', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n'],
 'Ġtechniques.': ['Ġ', 't', 'e', 'c', 'h', 'n', 'i', 'q', 'u', 'e', 's', '.']}
```
**Main loop: repeat until the vocabulary has reached a set size**
We compute the most frequent pairs, first by computing the pairs in a single word, then multiplying it by the frequency of this word and finally adding the counts of similar pairs together :

```python
# Top 10 most frequent pairs

('Ġ', 't'): 5
('t', 'o'): 3
('a', 'l'): 2
('i', 's'): 2
('u', 't'): 2
('e', 'n'): 2
('n', 'i'): 2
('T', 'h'): 1
('h', 'e'): 1
('Ġ', 'g'): 1
```

We add the most frequent pair to the vocabulary :

```python

vocabulary = ['u', 'h', 's', 'i', 't', 'q', 'o', 'r', 'T', 'l', 'n', 'f', 'c', '.', 'z', 'e', 'b', 'Ġ', 'd', 'a', 'k', 'g', 'Ġt']
```

We apply the merge rule to our splits:

```python
{'The': ['T', 'h', 'e'],
 'Ġgoal': ['Ġ', 'g', 'o', 'a', 'l'],
 'Ġof': ['Ġ', 'o', 'f'],
 'Ġthis': ['Ġt', 'h', 'i', 's'],
 'Ġtutorial': ['Ġt', 'u', 't', 'o', 'r', 'i', 'a', 'l'],
 'Ġis': ['Ġ', 'i', 's'],
 'Ġto': ['Ġt', 'o'],
 'Ġlearn': ['Ġ', 'l', 'e', 'a', 'r', 'n'],
 'Ġabout': ['Ġ', 'a', 'b', 'o', 'u', 't'],
 'Ġdifferent': ['Ġ', 'd', 'i', 'f', 'f', 'e', 'r', 'e', 'n', 't'],
 'Ġtokenization': ['Ġt', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n'],
 'Ġtechniques.': ['Ġt', 'e', 'c', 'h', 'n', 'i', 'q', 'u', 'e', 's', '.']}
```

**End of the loop**

Note that the original GPT-2 differs from our example as it works with the byte UTF-8 encoding.

[1] [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909)