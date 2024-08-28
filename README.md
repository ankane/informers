# Informers

:fire: Fast [transformer](https://github.com/xenova/transformers.js) inference for Ruby

For non-ONNX models, check out [Transformers.rb](https://github.com/ankane/transformers-ruby) :slightly_smiling_face:

[![Build Status](https://github.com/ankane/informers/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/informers/actions)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem "informers"
```

## Getting Started

- [Models](#models)
- [Pipelines](#pipelines)

## Models

### sentence-transformers/all-MiniLM-L6-v2

[Docs](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

```ruby
sentences = ["This is an example sentence", "Each sentence is converted"]

model = Informers::Model.new("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.embed(sentences)
```

For a quantized version, use:

```ruby
model = Informers::Model.new("Xenova/all-MiniLM-L6-v2", quantized: true)
```

### Xenova/multi-qa-MiniLM-L6-cos-v1

[Docs](https://huggingface.co/Xenova/multi-qa-MiniLM-L6-cos-v1)

```ruby
query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

model = Informers::Model.new("Xenova/multi-qa-MiniLM-L6-cos-v1")
query_embedding = model.embed(query)
doc_embeddings = model.embed(docs)
scores = doc_embeddings.map { |e| e.zip(query_embedding).sum { |d, q| d * q } }
doc_score_pairs = docs.zip(scores).sort_by { |d, s| -s }
```

### mixedbread-ai/mxbai-embed-large-v1

[Docs](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)

```ruby
def transform_query(query)
  "Represent this sentence for searching relevant passages: #{query}"
end

docs = [
  transform_query("puppy"),
  "The dog is barking",
  "The cat is purring"
]

model = Informers::Model.new("mixedbread-ai/mxbai-embed-large-v1")
embeddings = model.embed(docs)
```

### Supabase/gte-small

[Docs](https://huggingface.co/Supabase/gte-small) [unreleased]

```ruby
sentences = ["That is a happy person", "That is a very happy person"]

model = Informers::Model.new("Supabase/gte-small")
embeddings = model.embed(sentences)
```

## Pipelines

Named-entity recognition

```ruby
ner = Informers.pipeline("ner")
ner.("Ruby is a programming language created by Matz")
```

Sentiment analysis

```ruby
classifier = Informers.pipeline("sentiment-analysis")
classifier.("We are very happy to show you the 🤗 Transformers library.")
```

Question answering

```ruby
qa = Informers.pipeline("question-answering")
qa.("Who invented Ruby?", "Ruby is a programming language created by Matz")
```

Feature extraction

```ruby
extractor = Informers.pipeline("feature-extraction")
extractor.("We are very happy to show you the 🤗 Transformers library.")
```

## Credits

This library was ported from [Transformers.js](https://github.com/xenova/transformers.js) and is available under the same license.

## Upgrading

### 1.0

Task classes have been replaced with the `pipeline` method.

```ruby
# before
model = Informers::SentimentAnalysis.new("sentiment-analysis.onnx")
model.predict("This is super cool")

# after
model = Informers.pipeline("sentiment-analysis")
model.("This is super cool")
```

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
bundle exec rake test
```
