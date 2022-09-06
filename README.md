# Informers

:slightly_smiling_face: State-of-the-art natural language processing for Ruby

Supports:

- Sentiment analysis
- Question answering
- Named-entity recognition
- Text generation

[![Build Status](https://github.com/ankane/informers/workflows/build/badge.svg?branch=master)](https://github.com/ankane/informers/actions)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem "informers"
```

## Getting Started

- [Sentiment analysis](#sentiment-analysis)
- [Question answering](#question-answering)
- [Named-entity recognition](#named-entity-recognition)
- [Text generation](#text-generation)
- [Feature extraction](#feature-extraction)
- [Fill mask](#fill-mask)

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

First, download the [pretrained model](https://github.com/ankane/informers/releases/download/v0.1.0/question-answering.onnx).

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

### Text Generation

First, export the [pretrained model](tools/export.md).

Pass a prompt

```ruby
model = Informers::TextGeneration.new("text-generation.onnx")
model.predict("As far as I am concerned, I will", max_length: 50)
```

This returns

```text
As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a "free market." I think that the idea of a free market is a bit of a stretch. I think that the idea
```

### Feature Extraction

First, export a [pretrained model](tools/export.md).

```ruby
model = Informers::FeatureExtraction.new("feature-extraction.onnx")
model.predict("This is super cool")
```

### Fill Mask

First, export a [pretrained model](tools/export.md).

```ruby
model = Informers::FillMask.new("fill-mask.onnx")
model.predict("This is a great <mask>")
```

## Models

Task | Description | Contributor | License | Link
--- | --- | --- | --- | ---
Sentiment analysis | DistilBERT fine-tuned on SST-2 | Hugging Face | Apache-2.0 | [Link](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
Question answering | DistilBERT fine-tuned on SQuAD | Hugging Face | Apache-2.0 | [Link](https://huggingface.co/distilbert-base-cased-distilled-squad)
Named-entity recognition | BERT fine-tuned on CoNLL03 | Bayerische Staatsbibliothek | In-progress | [Link](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english)
Text generation | GPT-2 | OpenAI | [Custom](https://github.com/openai/gpt-2/blob/master/LICENSE) | [Link](https://huggingface.co/gpt2)

Some models are [quantized](https://medium.com/microsoftazure/faster-and-smaller-quantized-nlp-with-hugging-face-and-onnx-runtime-ec5525473bb7) to make them faster and smaller.

## Deployment

Check out [Trove](https://github.com/ankane/trove) for deploying models.

```sh
trove push sentiment-analysis.onnx
```

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
