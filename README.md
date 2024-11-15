# Informers

:fire: Fast [transformer](https://github.com/huggingface/transformers.js) inference for Ruby

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
- [sentence-transformers/multi-qa-MiniLM-L6-cos-v1](#sentence-transformersmulti-qa-MiniLM-L6-cos-v1)
- [sentence-transformers/all-mpnet-base-v2](#sentence-transformersall-mpnet-base-v2)
- [sentence-transformers/paraphrase-MiniLM-L6-v2](#sentence-transformersparaphrase-minilm-l6-v2)
- [mixedbread-ai/mxbai-embed-large-v1](#mixedbread-aimxbai-embed-large-v1)
- [Supabase/gte-small](#supabasegte-small)
- [intfloat/e5-base-v2](#intfloate5-base-v2)
- [nomic-ai/nomic-embed-text-v1](#nomic-ainomic-embed-text-v1)
- [BAAI/bge-base-en-v1.5](#baaibge-base-en-v15)
- [jinaai/jina-embeddings-v2-base-en](#jinaaijina-embeddings-v2-base-en)
- [Snowflake/snowflake-arctic-embed-m-v1.5](#snowflakesnowflake-arctic-embed-m-v15)

Reranking

- [mixedbread-ai/mxbai-rerank-base-v1](#mixedbread-aimxbai-rerank-base-v1)
- [jinaai/jina-reranker-v1-turbo-en](#jinaaijina-reranker-v1-turbo-en)
- [BAAI/bge-reranker-base](#baaibge-reranker-base)
- [Xenova/ms-marco-MiniLM-L-6-v2](#xenovams-marco-minilm-l-6-v2)

### sentence-transformers/all-MiniLM-L6-v2

[Docs](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

```ruby
sentences = ["This is an example sentence", "Each sentence is converted"]

model = Informers.pipeline("embedding", "sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.(sentences)
```

### sentence-transformers/multi-qa-MiniLM-L6-cos-v1

[Docs](https://huggingface.co/Xenova/multi-qa-MiniLM-L6-cos-v1)

```ruby
query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

model = Informers.pipeline("embedding", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
query_embedding = model.(query)
doc_embeddings = model.(docs)
scores = doc_embeddings.map { |e| e.zip(query_embedding).sum { |d, q| d * q } }
doc_score_pairs = docs.zip(scores).sort_by { |d, s| -s }
```

### sentence-transformers/all-mpnet-base-v2

[Docs](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

```ruby
sentences = ["This is an example sentence", "Each sentence is converted"]

model = Informers.pipeline("embedding", "sentence-transformers/all-mpnet-base-v2")
embeddings = model.(sentences)
```

### sentence-transformers/paraphrase-MiniLM-L6-v2

[Docs](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)

```ruby
sentences = ["This is an example sentence", "Each sentence is converted"]

model = Informers.pipeline("embedding", "sentence-transformers/paraphrase-MiniLM-L6-v2")
embeddings = model.(sentences, normalize: false)
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

### Xenova/ms-marco-MiniLM-L-6-v2

[Docs](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2)

```ruby
query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

model = Informers.pipeline("reranking", "Xenova/ms-marco-MiniLM-L-6-v2")
result = model.(query, docs)
```

### Other

The model must include a `.onnx` file ([example](https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main/onnx)). If the file is not at `onnx/model.onnx`, use the `model_file_name` option to specify the location.

## Pipelines

- [Text](#text)
- [Vision](#vision)
- [Audio](#audio)
- [Multimodel](#multimodal)

### Text

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

Zero-shot classification

```ruby
classifier = Informers.pipeline("zero-shot-classification")
classifier.("text", ["label1", "label2", "label3"])
```

Text generation

```ruby
generator = Informers.pipeline("text-generation")
generator.("I enjoy walking with my cute dog,")
```

Text-to-text generation

```ruby
text2text = Informers.pipeline("text2text-generation")
text2text.("translate from English to French: I'm very happy")
```

Translation

```ruby
translator = Informers.pipeline("translation", "Xenova/nllb-200-distilled-600M")
translator.("जीवन एक चॉकलेट बॉक्स की तरह है।", src_lang: "hin_Deva", tgt_lang: "fra_Latn")
```

Summarization

```ruby
summarizer = Informers.pipeline("summarization")
summarizer.("Many paragraphs of text")
```

Fill mask

```ruby
unmasker = Informers.pipeline("fill-mask")
unmasker.("Paris is the [MASK] of France.")
```

Feature extraction

```ruby
extractor = Informers.pipeline("feature-extraction")
extractor.("We are very happy to show you the 🤗 Transformers library.")
```

### Vision

Note: [ruby-vips](https://github.com/libvips/ruby-vips) is required to load images

Image classification

```ruby
classifier = Informers.pipeline("image-classification")
classifier.("image.jpg")
```

Zero-shot image classification

```ruby
classifier = Informers.pipeline("zero-shot-image-classification")
classifier.("image.jpg", ["label1", "label2", "label3"])
```

Image segmentation

```ruby
segmenter = Informers.pipeline("image-segmentation")
segmenter.("image.jpg")
```

Object detection

```ruby
detector = Informers.pipeline("object-detection")
detector.("image.jpg")
```

Zero-shot object detection

```ruby
detector = Informers.pipeline("zero-shot-object-detection")
detector.("image.jpg", ["label1", "label2", "label3"])
```

Depth estimation

```ruby
estimator = Informers.pipeline("depth-estimation")
estimator.("image.jpg")
```

Image-to-image

```ruby
upscaler = Informers.pipeline("image-to-image")
upscaler.("image.jpg")
```

Image feature extraction

```ruby
extractor = Informers.pipeline("image-feature-extraction")
extractor.("image.jpg")
```

### Audio

Note: [ffmpeg](https://www.ffmpeg.org/) is required to load audio files

Audio classification

```ruby
classifier = Informers.pipeline("audio-classification")
classifier.("audio.wav")
```

### Multimodal

Image captioning

```ruby
captioner = Informers.pipeline("image-to-text")
captioner.("image.jpg")
```

Document question answering

```ruby
qa = Informers.pipeline("document-question-answering")
qa.("image.jpg", "What is the invoice number?")
```

## Reference

Specify a variant of the model if available (`fp32`, `fp16`, `int8`, `uint8`, `q8`, `q4`, `q4f16`, or `bnb4`)

```ruby
Informers.pipeline("embedding", "Xenova/all-MiniLM-L6-v2", dtype: "fp16")
```

Specify a device (`cpu`, `cuda`, or `coreml`)

```ruby
Informers.pipeline("embedding", device: "cuda")
```

Note: Follow [these instructions](https://github.com/ankane/onnxruntime-ruby?tab=readme-ov-file#gpu-support) for `cuda`

Specify ONNX Runtime [session options](https://github.com/ankane/onnxruntime-ruby?tab=readme-ov-file#session-options)

```ruby
Informers.pipeline("embedding", session_options: {log_severity_level: 2})
```

## Credits

This library was ported from [Transformers.js](https://github.com/huggingface/transformers.js) and is available under the same license.

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
bundle exec rake download:files
bundle exec rake test
```
