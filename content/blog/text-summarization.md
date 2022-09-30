---
author: "Vien Vuong"
title: "Text Summarization Methods in NLP"
date: "2022-03-16"
description: "Summarization is the task of producing a shorter version of one or several documents that that represents the most important or relevant information within the original content. This guide goes over the state-of-the-art text summarization techniques in natural language processing (as of early 2020)."
tags: ["text-summarization", "nlp", "ml"]
comments: false
socialShare: false
toc: true
cover:
  src: /text-summarization/cover.png
  alt: Text Summarization
---

Summarization is the task of producing a shorter version of one or several documents that that represents the most important or relevant information within the original content. This guide goes over the state-of-the-art text summarization techniques in natural language processing (as of early 2020).

## Can we summarize a document (a piece of text) to a more concise text?

Humans can quite trivially summarize a document after reading it. However, this is not the case for computers as they lack human knowledge and language capability to develop an understanding of the text. While summarization remains very challenging, there have been a lot of ground-breaking research that produces impressive results and pushes the boundary of language comprehension. We will briefly explore the metrics used to evaluate the quality of text summarization, the different approaches to the summarization task, and the current state-of-the-art models and results.

## Evaluation Metrics

1. **ROUGE-N**: Word n-gram count of matches between the model and the gold summary. It is a generalization of "recall" because it evaluates the proportion of words in the gold summary captured by the candidate summary, and extra n-grams in the candidate summary do not affect the score.
   - ROUGE-1 and ROUGE-2 is often used.
   - The ROUGE-1 means word base, so its order is not regarded. So "apple bee" and "bee apple" has the same ROUGE-1 score.
   - But if ROUGE-2, "apple bee" becomes single entity so "apple bee" and "bee apple" does not match.
   - If you increase the ROUGE-"N" count, finally evaluates completely match or not.
   - ROUGE-L is also often used. It measures the Longest Common Subsequence (LCS) words between reference and candidate summaries. By LCS, we refer to word tokens that are in sequence, but not necessarily consecutive.
   - There are precision/recall variants for ROUGE-N score, though the recall score is usually the default.
   - F1-score is the harmonic mean of precision and recall. It can give a reliable measure of a model performance that relies not only on the model capturing as many words as possible (recall) but doing so without outputting irrelevant words (precision).
   - In general, ROUGE-1 precision/recall is almost the same as BLEU precision/recall save for BLEU's brevity penalty.
2. **BLEU**: A generalization of "precision" widely used in machine translation evaluation. BLEU is calculated on the n-gram co-occerance between the generated summary and the gold (no need to specify the "n" unlike ROUGE).
   - BLEU introduces a brevity penalty that penalizes the candidate summary for being shorter than the reference summary.
   - There are precision/recall variants for BLEU score, though the precision score is usually the default.

- **Warning**: For summarization, automatic metrics such as ROUGE-N and BLUE have serious limitations:
  1. They only assess content selection and do not account for other quality aspects, such as fluency, grammaticality, coherence, etc.
  2. To assess content selection, they rely mostly on lexical overlap, although an abstractive summary could express they same content as a reference without any lexical overlap.
  3. Given the subjectiveness of summarization and the correspondingly low agreement between annotators, the metrics were designed to be used with multiple reference summaries per input. However, recent datasets such as CNN/DailyMail and Gigaword provide only a single reference.
- Therefore, tracking progress and claiming state-of-the-art based only on these metrics is questionable. Most papers carry out additional manual comparisons of alternative summaries. Unfortunately, such experiments are difficult to compare across papers.

## What are the techniques used to summarize text, and how do they work?

### Two (basic) types of text summarization

Automatic Text Summarization training is usually a supervised learning process, where the target for each text passage is a corresponding golden annotated summary (human-expert guided summary). On the other hand, the model generated summaries or predictions after training are of 2 types:

1. **Extractive Summarization**: The summaries contain wordings and phrases extracted from the original passage. The most common method is to extract sentences in the right order from the original text while favoring the first few sentences as they are usually the most important.

   - Pros: Summarization is quite robust you can rely on the extracted sentences to be grammatically correct and coherent.
   - Cons: Lacks flexibility as you cannot use novel words or connectors. Also unintuitive and verbose as most people paraphrase to best summarize a text.
   - Categories:
     - Graph-based: Generates a graph from the document, then summarizes it by considering the relation between the nodes (text-unit).
       - Examples: [TextRank](https://aclanthology.org/W04-3252) (gensim summarization.summarizer)
     - Feature-based: Extracts the features of a sentence then evaluate its importance. The features include the sentence's position in document, presence of verbs, length, term frequency, named entity tags, and font style.
       - Examples: Luhn Algorithm, [TextTeaser](https://github.com/IndigoResearch/textteaser), [SummaRuNNer](https://arxiv.org/abs/1611.04230)
     - Topic-based: Finds the topic of the document and evaluate each sentences by what kinds of topics are included (the "main" topic is ranked highly when scoring the sentence). Latent Semantic Analysis (LSA), an SVD-based technique, is usually used to detect the topic.
       - Examples: LSA-based topic models (gensim models.lsimodel)
     - Grammar-based: Parses the text and constructs a grammatical structure, then select/reorder the substructures.
     - Neural-network-based: Uses a neural network to generate good sentence representation (encoding), and to predict the selection of sentence (objective function).

2. **Abstractive Summarization**: The summaries are completely paraphrased, sometimes with words and content that might not occur in the original passage.
   - Pros: Intuitive, summarizes like how humans do. Flexible as you can use words not in the original text. Allows for more fluent and natural summaries.
   - Cons: It is a much more difficult task, as the model has to generate summaries that have coherent phrases and connectors.
   - Encoder-decoder model: The encoder converts an input document to a latent representation (vector), and the decoder generates a summary using said latent representation.
     - [A Neural Attention Model for Abstractive Sentence Summarization](https://aclanthology.org/D15-1044) (Rush et al., EMNLP 2015)
     - [Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](https://aclanthology.org/N16-1012) (Chopra et al., NAACL 2016)
     - [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://aclanthology.org/K16-1028) (Nallapati et al., CoNLL 2016)

### Combination Approach

The model generates a summary that is a combination of both extractive and abstractive summarization.

- Pros: It is a compromise between the two approaches.
- Cons: It is still difficult to generate a coherent summary.

1. **Pointer-Generator Network**: Combine the extractive and abstractive model by switching probability.
   - [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304) (Paulus et al., arXiv 2017)
   - [Controlling Decoding for More Abstractive Summaries with Copy-Based Networks](https://arxiv.org/abs/1803.07038) (Weber et al., arXiv 2018)
2. **Extract-then-Abstract Model**: Use extractive model to select the sentence from documents, then adopt the abstractive model to selected sentences.
   - [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198) (Liu et al., arXiv 2018)
   - [Query Focused Abstractive Summarization: Incorporating Query Relevance, Multi-Document Coverage, and Summary Length Constraints into seq2seq Models](https://arxiv.org/abs/1801.07704) (Baumel et al., arXiv 2018)
   - [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://aclanthology.org/P18-1063) (Chen & Bansal, ACL 2018)
3. **Transfer Learning**: Allows reusing a learned model to make a new summarization model, and as a result, reduces the amount of training data and time required. Can train domain-specific summarization models with little data and/or within a short amount of time.
   - [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://aclanthology.org/P18-1063) (Chen & Bansal, ACL 2018)
   - [Pretraining-Based Natural Language Generation for Text Summarization](https://aclanthology.org/K19-1074) (Zhang et al., CoNLL 2019)
   - [Fine-tune BERT for Extractive Summarization](https://arxiv.org/abs/1903.10318) (Liu et al., arXiv 2019)

### State of the Art

1. Pre-Transformers: before the 2018 paper, [BERT](https://aclanthology.org/N19-1423), LSTMs (with and without attention mechanism) are considered SOTA.
   - [Neural Summarization by Extracting Sentences and Words](https://aclanthology.org/P16-1046) (Cheng & Lapata, ACL 2016)
2. Post-Transformers: transformer-based models dominate the state of the art.
   - [Text Summarization with Pretrained Encoders](https://aclanthology.org/D19-1387) (Liu & Lapata, EMNLP 2019)
   - [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](http://proceedings.mlr.press/v119/zhang20ae/zhang20ae.pdf) (Zhang et al., ICML 2020)

### Some Seminal Papers

1. [The Automatic Creation of Literature Abstracts](https://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf) (Luhn, 1958)

   - Ignore stopwords: determiners (the, a, an), coordinating conjunctions (for, and, or, but), and prepositions (in, under, before) are ignored.
   - Determine top words: the most often occuring words in the document are counted up.
   - Select top words: A small number of the top words are selected to be used for scoring.
   - Select top sentences: Sentences are scored according to how many of the top words they contain. The top N sentences are selected for the summary.

2. [TextRank: Bringing Order into Text](https://aclanthology.org/W04-3252) (Mihalcea & Tarau, 2004)

   - Based on the Google's PageRank algorithm.
   - The text is split into sentences, which are then each converted into a vector representation (word embeddings).
   - Similarities between sentence vectors are then calculated and stored in a matrix.
   - The similarity matrix is then converted into a graph, with sentences as vertices and similarity scores as edges, for sentence rank calculation.
   - Finally, run the PageRank algorithm on the graph and select the N top-ranked sentences (vertices) to form the summary

3. [A Neural Attention Model for Abstractive Sentence Summarization](https://aclanthology.org/D15-1044) (Rush et al., EMNLP 2015)

   - Unsupervised seq2seq encoder-decoder LSTM with attention.
   - Use the first sentence of a document.
   - Set focus on the important sentences and keywords by using word and sentence level attention.
   - Handle ovel/rare but important words by adding an n-gram match term to the loss function.
   - Capture the local context by using 1D convolution
   - Use beam search to generate the summary.
   - The source document is quite small (about 1 paragraph or ~500 words in the training dataset of Gigaword) and the produced output is also very short (about 75 characters).
   - It remains an open challenge to scale up these limits - to produce longer summaries over multi-paragraph text input (even good LSTM models with attention models fall victim to vanishing gradients when the input sequences become longer than a few hundred items).

4. [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://aclanthology.org/K16-1028) (Nallapati et al., CoNLL 2016)

   - Unsupervised encoder-decoder seq2seq bidirectional GRU with attention.
   - Use the first 2 sentences of a documnet with a limit at 120 words.
   - Set focus on the important sentences and keywords by using word and sentence level attention and enhanced features. Features such as TF-IDF scores, part-of-speech (POS) tags (NOUN, VERB, ADJ, etc.), and Named Entity types (Person, Location, Event, etc.) are added (concatenated) to the encodings of the words, which adds additional encoding dimensions that reflect the context and importance of the words.
   - Use the [Large Vocabulary Trick (LVT)](https://arxiv.org/abs/1412.2007) (Jean et al. 2014) to reduce perplexity. This means when you decode, use only the words that appear in the source.
   - To prevent the summary from being too extractive, "vocabulary expansion" is used, i.e., a layer of word2vec nearest-neighbor embeddings is added to the words in the input.
   - Most importantly, a "Switching Pointer-Generator" layer is added to the decoder. This layer decides whether to generate a new word based on the context / previously generated word (as a generator and typical decoder), or to copy a word from the input (as a pointer).

5. [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](http://proceedings.mlr.press/v119/zhang20ae/zhang20ae.pdf) (Zhang et al., ICML 2020)

   - One of the current SOTA models.
   - Self-supervised encoder-decoder seq2seq transformer with attention.
   - As for most transfomer-based summarization models, there is a crucial pre-training step. One common paradigm is self-supervised masked language modeling: letting the model iteratively predict masked out tokens (words) from an input document. PEGASUS chooses to instead mask out whole sentences from the input, which makes this training task equivalent to extractive summarization. This is referred to as gap-sentences generation in the paper.
   - Inspired by [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://paperswithcode.com/paper/exploring-the-limits-of-transfer-learning) (Raffel et al.): masking multiple tokens in the pre-training step.
   - Datasets used for pre-training are T5's 750GB C4 (Colossal and Cleaned version of Common Crawl), and PEGASUS' own HugeNews corpus which contains 3.8TB web-crawled news articles.
   - Once pre-training is done, you would fine tune that model on a dataset of labelled summaries (XSum, CNN/Dailymail, Gigaword, etc.) and hope it generalizes to that collection of documents and writes an abstractive novel summary that it hasn't seen in the pre-training set.
   - Largest improvement achieved in zero-shot and low-resource summarization (approaches SOTA with much fewer training examples), which allows for fine-tuning without the need for massive labeled datasets.

6. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://aclanthology.org/2020.acl-main.703) (Lewis et al., ACL 2020)

   - One of the current SOTA models.
   - Self-supervised encoder-decoder seq2seq Bidirectional Autoregressive TRansformer with attention.
   - Use a bidirectional encoder (like [BERT](https://aclanthology.org/N19-1423)), and an auto-regressive (left-to-right) decoder (like [GPT-1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)).
   - Very similar pre-training approach to PEGASUS. The noising encoding scheme used includes token masking, token deletion, text infilling, sentence permutation, and document rotation.
   - Use Wikipedia and BookCorpus datasets for pre-training, totalling around 160GB.
   - Similar to PEGASUS, BART can also be fine-tuned over any text summarization datasets (XSum, CNN/Dailymail, Gigaword, etc.).

## Sources

- Anon, 2022. [Papers with Code - Text Summarization](https://paperswithcode.com/task/text-summarization). Paperswithcode.com.

- Gonçalves, L., 2020. [Automatic Text Summarization with Machine Learning  —  An overview](https://towardsdatascience.com/introduction-to-text-summarization-with-rouge-scores-84140c64b471). Medium.
- Briggs, J., 2021. [The Ultimate Performance Metric in NLP](https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460). Medium.
- Tan, A., 2022. [Introduction to Text Summarization with ROUGE Scores](https://towardsdatascience.com/introduction-to-text-summarization-with-rouge-scores-84140c64b471). Medium.
- Pietro, M., 2022. [Text Summarization with NLP: TextRank vs Seq2Seq vs BART](https://towardsdatascience.com/text-summarization-with-nlp-textrank-vs-seq2seq-vs-bart-474943efeb09). Medium.

- Kubo, T. and Mamdapure, A., 2022. [icoxfog417/awesome-text-summarization: The guide to tackle with the Text Summarization](https://github.com/icoxfog417/awesome-text-summarization#basic-approach). GitHub.
- Anon, 2021. [NLP-progress/summarization.md at master · sebastianruder/NLP-progress](https://github.com/sebastianruder/NLP-progress/blob/master/english/summarization.md). GitHub.

- [TextRank: Bringing Order into Text](https://aclanthology.org/W04-3252) (Mihalcea & Tarau, 2004)
- [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230) (Nallapati et al., 2016)
- [Neural Summarization by Extracting Sentences and Words](https://aclanthology.org/P16-1046) (Cheng & Lapata, ACL 2016)
- [A Neural Attention Model for Abstractive Sentence Summarization](https://aclanthology.org/D15-1044) (Rush et al., EMNLP 2015)
- [Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](https://aclanthology.org/N16-1012) (Chopra et al., NAACL 2016)
- [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://aclanthology.org/K16-1028) (Nallapati et al., CoNLL 2016)
- [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304) (Paulus et al., arXiv 2017)
- [Controlling Decoding for More Abstractive Summaries with Copy-Based Networks](https://arxiv.org/abs/1803.07038) (Weber et al., arXiv 2018)
- [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198) (Liu et al., arXiv 2018)
- [Query Focused Abstractive Summarization: Incorporating Query Relevance, Multi-Document Coverage, and Summary Length Constraints into seq2seq Models](https://arxiv.org/abs/1801.07704) (Baumel et al., arXiv 2018)
- [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://aclanthology.org/P18-1063) (Chen & Bansal, ACL 2018)
- [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://aclanthology.org/P18-1063) (Chen & Bansal, ACL 2018)
- [Pretraining-Based Natural Language Generation for Text Summarization](https://aclanthology.org/K19-1074) (Zhang et al., CoNLL 2019)
- [Fine-tune BERT for Extractive Summarization](https://arxiv.org/abs/1903.10318) (Liu et al., arXiv 2019)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423) (Devlin et al., NAACL 2019)
- [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., Preprint 2018)
- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://aclanthology.org/2020.acl-main.703) (Lewis et al., ACL 2020)
- [Text Summarization with Pretrained Encoders](https://aclanthology.org/D19-1387) (Liu & Lapata, EMNLP 2019)
- [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](http://proceedings.mlr.press/v119/zhang20ae/zhang20ae.pdf) (Zhang et al., ICML 2020)
