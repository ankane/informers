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

Embedding

- [sentence-transformers/all-MiniLM-L6-v2](#sentence-transformersall-MiniLM-L6-v2)
- [Xenova/multi-qa-MiniLM-L6-cos-v1](#xenovamulti-qa-MiniLM-L6-cos-v1)
- [mixedbread-ai/mxbai-embed-large-v1](#mixedbread-aimxbai-embed-large-v1)
- [Supabase/gte-small](#supabasegte-small)
- [intfloat/e5-base-v2](#intfloate5-base-v2)
- [nomic-ai/nomic-embed-text-v1](#nomic-ainomic-embed-text-v1)
- [BAAI/bge-base-en-v1.5](#baaibge-base-en-v15)
- [jinaai/jina-embeddings-v2-base-en](#jinaaijina-embeddings-v2-base-en)
- [Snowflake/snowflake-arctic-embed-m-v1.5](#snowflakesnowflake-arctic-embed-m-v15)
- [Xenova/all-mpnet-base-v2](#xenovaall-mpnet-base-v2)

Reranking

- [mixedbread-ai/mxbai-rerank-base-v1](#mixedbread-aimxbai-rerank-base-v1)
- [jinaai/jina-reranker-v1-turbo-en](#jinaaijina-reranker-v1-turbo-en)
- [BAAI/bge-reranker-base](#baaibge-reranker-base)

### sentence-transformers/all-MiniLM-L6-v2

[Docs](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

```ruby
sentences = ["This is an example sentence", "Each sentence is converted"]

model = Informers.pipeline("embedding", "sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.(sentences)
```

For a quantized version, use:

```ruby
model = Informers.pipeline("embedding", "Xenova/all-MiniLM-L6-v2", quantized: true)
```

### Xenova/multi-qa-MiniLM-L6-cos-v1

[Docs](https://huggingface.co/Xenova/multi-qa-MiniLM-L6-cos-v1)

```ruby
query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

model = Informers.pipeline("embedding", "Xenova/multi-qa-MiniLM-L6-cos-v1")
query_embedding = model.(query)
doc_embeddings = model.(docs)
scores = doc_embeddings.map { |e| e.zip(query_embedding).sum { |d, q| d * q } }
doc_score_pairs = docs.zip(scores).sort_by { |d, s| -s }
```

### mixedbread-ai/mxbai-embed-large-v1

[Docs](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)

```ruby
query_prefix = "Represent this sentence for searching relevant passages: "

input = [
  "The dog is barking",
  "The cat is purring",
  query_prefix + "puppy"
]

model = Informers.pipeline("embedding", "mixedbread-ai/mxbai-embed-large-v1")
embeddings = model.(input)
```

### Supabase/gte-small

[Docs](https://huggingface.co/Supabase/gte-small)

```ruby
sentences = ["That is a happy person", "That is a very happy person"]

model = Informers.pipeline("embedding", "Supabase/gte-small")
embeddings = model.(sentences)
```

### intfloat/e5-base-v2

[Docs](https://huggingface.co/intfloat/e5-base-v2)

```ruby
doc_prefix = "passage: "
query_prefix = "query: "

input = [
  doc_prefix + "Ruby is a programming language created by Matz",
  query_prefix + "Ruby creator"
]

model = Informers.pipeline("embedding", "intfloat/e5-base-v2")
embeddings = model.(input)
```

### nomic-ai/nomic-embed-text-v1

[Docs](https://huggingface.co/nomic-ai/nomic-embed-text-v1)

```ruby
doc_prefix = "search_document: "
query_prefix = "search_query: "

input = [
  doc_prefix + "The dog is barking",
  doc_prefix + "The cat is purring",
  query_prefix + "puppy"
]

model = Informers.pipeline("embedding", "nomic-ai/nomic-embed-text-v1")
embeddings = model.(input)
```

### BAAI/bge-base-en-v1.5

[Docs](https://huggingface.co/BAAI/bge-base-en-v1.5)

```ruby
query_prefix = "Represent this sentence for searching relevant passages: "

input = [
  "The dog is barking",
  "The cat is purring",
  query_prefix + "puppy"
]

model = Informers.pipeline("embedding", "BAAI/bge-base-en-v1.5")
embeddings = model.(input)
```

### jinaai/jina-embeddings-v2-base-en

[Docs](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)

```ruby
sentences = ["How is the weather today?", "What is the current weather like today?"]

model = Informers.pipeline("embedding", "jinaai/jina-embeddings-v2-base-en", model_file_name: "../model")
embeddings = model.(sentences)
```

### Snowflake/snowflake-arctic-embed-m-v1.5

[Docs](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5)

```ruby
query_prefix = "Represent this sentence for searching relevant passages: "

input = [
  "The dog is barking",
  "The cat is purring",
  query_prefix + "puppy"
]

model = Informers.pipeline("embedding", "Snowflake/snowflake-arctic-embed-m-v1.5")
embeddings = model.(input, model_output: "sentence_embedding", pooling: "none")
```

### Xenova/all-mpnet-base-v2

[Docs](https://huggingface.co/Xenova/all-mpnet-base-v2)

```ruby
sentences = ["This is an example sentence", "Each sentence is converted"]

model = Informers.pipeline("embedding", "Xenova/all-mpnet-base-v2")
embeddings = model.(sentences)
```

### mixedbread-ai/mxbai-rerank-base-v1

[Docs](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1)

```ruby
query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

model = Informers.pipeline("reranking", "mixedbread-ai/mxbai-rerank-base-v1")
result = model.(query, docs)
```

### jinaai/jina-reranker-v1-turbo-en

[Docs](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en)

```ruby
query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

model = Informers.pipeline("reranking", "jinaai/jina-reranker-v1-turbo-en")
result = model.(query, docs)
```

### BAAI/bge-reranker-base

[Docs](https://huggingface.co/BAAI/bge-reranker-base)

```ruby
query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

model = Informers.pipeline("reranking", "BAAI/bge-reranker-base")
result = model.(query, docs)
```

### Other

You can use the feature extraction pipeline directly.

```ruby
model = Informers.pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", quantized: false)
embeddings = model.(sentences, pooling: "mean", normalize: true)
```

The model must include a `.onnx` file ([example](https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main/onnx)). If the file is not at `onnx/model.onnx` or `onnx/model_quantized.onnx`, use the `model_file_name` option to specify the location.

## Pipelines

Embedding

```ruby
embed = Informers.pipeline("embedding")
embed.("We are very happy to show you the 🤗 Transformers library.")
```

Reranking

```ruby
rerank = Informers.pipeline("reranking")
rerank.("Who created Ruby?", ["Matz created Ruby", "Another doc"])
```

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

Fill mask [unreleased]

```ruby
unmasker = Informers.pipeline("fill-mask")
unmasker.("Paris is the [MASK] of France.")
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
