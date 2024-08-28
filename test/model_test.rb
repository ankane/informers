require_relative "test_helper"

class ModelTest < Minitest::Test
  def setup
    skip if ci?
  end

  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  def test_sentence_transformers
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = Informers::Model.new("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.embed(sentences)

    assert_elements_in_delta [0.067657, 0.063496, 0.048713], embeddings[0][..2]
    assert_elements_in_delta [0.086439, 0.10276, 0.0053946], embeddings[1][..2]
  end

  # https://huggingface.co/Xenova/all-MiniLM-L6-v2
  def test_xenova
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = Informers::Model.new("Xenova/all-MiniLM-L6-v2", quantized: true)
    embeddings = model.embed(sentences)

    assert_elements_in_delta [0.045927, 0.07328, 0.05401], embeddings[0][..2]
    assert_elements_in_delta [0.081881, 0.1076, -0.01324], embeddings[1][..2]
  end

  # https://huggingface.co/Xenova/multi-qa-MiniLM-L6-cos-v1
  def test_xenova2
    query = "How many people live in London?"
    docs = ["Around 9 Million people live in London", "London is known for its financial district"]

    model = Informers::Model.new("Xenova/multi-qa-MiniLM-L6-cos-v1")
    query_embedding = model.embed(query)
    doc_embeddings = model.embed(docs)
    scores = doc_embeddings.map { |e| e.zip(query_embedding).sum { |d, q| d * q } }
    doc_score_pairs = docs.zip(scores).sort_by { |d, s| -s }

    assert_equal "Around 9 Million people live in London", doc_score_pairs[0][0]
    assert_in_delta 0.9156, doc_score_pairs[0][1]
    assert_equal "London is known for its financial district", doc_score_pairs[1][0]
    assert_in_delta 0.4948, doc_score_pairs[1][1]
  end

  # https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
  def test_mixedbread
    transform_query = lambda do |query|
      "Represent this sentence for searching relevant passages: #{query}"
    end

    docs = [
      transform_query.("puppy"),
      "The dog is barking",
      "The cat is purring"
    ]

    model = Informers::Model.new("mixedbread-ai/mxbai-embed-large-v1")
    embeddings = model.embed(docs)

    assert_elements_in_delta [-0.00624076, 0.12864432, 0.5248165], embeddings[0][..2]
    assert_elements_in_delta [-0.61227727, 1.4060247, -0.04079155], embeddings[-1][..2]
  end

  # https://huggingface.co/Supabase/gte-small
  def test_supabase
    sentences = ["That is a happy person", "That is a very happy person"]

    model = Informers::Model.new("Supabase/gte-small")
    embeddings = model.embed(sentences)

    assert_elements_in_delta [-0.05316979, 0.01044252, 0.06194701], embeddings[0][..2]
    assert_elements_in_delta [-0.05246907, 0.03752426, 0.07344585], embeddings[-1][..2]
  end

  # https://huggingface.co/intfloat/e5-base-v2
  def test_intfloat
    input = [
      "passage: Ruby is a programming language created by Matz",
      "query: Ruby creator"
    ]

    model = Informers::Model.new("intfloat/e5-base-v2")
    embeddings = model.embed(input)

    assert_elements_in_delta [-0.00596662, -0.03730119, -0.0703470], embeddings[0][..2]
    assert_elements_in_delta [0.00298353, -0.04421991, -0.0591884], embeddings[-1][..2]
  end

  # https://huggingface.co/nomic-ai/nomic-embed-text-v1
  def test_nomic
    input = [
      "search_document: The dog is barking",
      "search_query: puppy"
    ]

    model = Informers::Model.new("nomic-ai/nomic-embed-text-v1")
    embeddings = model.embed(input)

    assert_elements_in_delta [-0.00645858, 0.01145126, 0.0099767], embeddings[0][..2]
    assert_elements_in_delta [-0.01173127, 0.04957652, -0.0176401], embeddings[-1][..2]
  end
end
