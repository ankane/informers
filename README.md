# Informers

:slightly_smiling_face: State-of-the-art natural language processing for Ruby

Supports:

- Sentiment analysis
- Question answering
- Named-entity recognition
- Zero-shot classification
- Text generation - *in development*
- Summarization - *in development*
- Translation - *in development*

[![Build Status](https://travis-ci.org/ankane/informers.svg?branch=master)](https://travis-ci.org/ankane/informers)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem 'informers'
```

On Mac, also install OpenMP:

```sh
brew install libomp
```

## Getting Started

- [Sentiment analysis](#sentiment-analysis)
- [Question answering](#question-answering)
- [Named-entity recognition](#named-entity-recognition)
- [Zero-shot classification](#zero-shot-classification)

### Sentiment Analysis

First, download the [pretrained model](https://github.com/ankane/informers/releases/download/v0.1.0/sentiment-analysis.onnx).

Predict sentiment

```ruby
model = Informers::SentimentAnalysis.new("sentiment-analysis.onnx")
model.predict("This is super cool")
```

This returns

```ruby
{label: "positive", score: 0.999855186578301}
```

Predict multiple at once

```ruby
model.predict(["This is super cool", "I didn't like it"])
```

### Question Answering

First, download the [pretrained model](https://github.com/ankane/informers/releases/download/v0.1.0/question-answering.onnx) and add Numo to your application’s Gemfile:

```ruby
gem 'numo-narray'
```

Ask a question with some context

```ruby
model = Informers::QuestionAnswering.new("question-answering.onnx")
model.predict(
  question: "Who invented Ruby?",
  context: "Ruby is a programming language created by Matz"
)
```

This returns

```ruby
{answer: "Matz", score: 0.9980658360049758, start: 42, end: 46}
```

### Named-Entity Recognition

First, export the [pretrained model](tools/export.md).

Get entities

```ruby
model = Informers::NER.new("ner.onnx")
model.predict("Nat works at GitHub in San Francisco")
```

This returns

```ruby
[
  {text: "Nat",           tag: "person",   score: 0.9840519576513487, start: 0,  end: 3},
  {text: "GitHub",        tag: "org",      score: 0.9426134775785775, start: 13, end: 19},
  {text: "San Francisco", tag: "location", score: 0.9952414982243061, start: 23, end: 36}
]
```

### Zero-Shot Classification

First, download the [pretrained model](https://github.com/ankane/informers/releases/download/v0.1.0/zero-shot-classification.onnx).

Get entities

```ruby
model = Informers::ZeroShotClassification.new("zero-shot-classification.onnx")
model.predict("Who are you voting for in 2020?", ["Europe", "public health", "politics"])
```

This returns the score for each label

```ruby
[123, 123, 123]
```

## Models

Task | Description | Contributor | License | Link
--- | --- | --- | --- | ---
Sentiment analysis | DistilBERT fine-tuned on SST-2 | Hugging Face | Apache-2.0 | [Link](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
Question answering | DistilBERT | Hugging Face | Apache-2.0 | [Link](https://huggingface.co/distilbert-base-cased-distilled-squad)
Named-entity recognition | BERT fine-tuned on CoNLL03 | Bayerische Staatsbibliothek | In-progress | [Link](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english)
Zero-shot classification | BART | Facebook | MIT | [Link](https://huggingface.co/facebook/bart-large-mnli)

Models are [quantized](https://medium.com/microsoftazure/faster-and-smaller-quantized-nlp-with-hugging-face-and-onnx-runtime-ec5525473bb7) to make them faster and smaller.

## Credits

This project uses many state-of-the-art technologies:

- [Transformers](https://github.com/huggingface/transformers) for transformer models
- [Bling Fire](https://github.com/microsoft/BlingFire) and [BERT](https://github.com/google-research/bert) for high-performance text tokenization
- [ONNX Runtime](https://github.com/Microsoft/onnxruntime) for high-performance inference

Some code was ported from Transformers and is available under the same license.

## History

View the [changelog](https://github.com/ankane/informers/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/informers/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/informers/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/informers.git
cd informers
bundle install

export MODELS_PATH=path/to/onnx/models
bundle exec rake test
```
