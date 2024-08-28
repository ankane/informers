require_relative "test_helper"

class ModelTest < Minitest::Test
  def setup
    skip if ci?
  end

  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  def test_all_minilm
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = Informers.pipeline("embedding", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.(sentences)

    assert_elements_in_delta [0.067657, 0.063496, 0.048713], embeddings[0][..2]
    assert_elements_in_delta [0.086439, 0.10276, 0.0053946], embeddings[1][..2]
  end

  # https://huggingface.co/Xenova/all-MiniLM-L6-v2
  def test_all_minilm_xenova
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = Informers.pipeline("embedding", "Xenova/all-MiniLM-L6-v2", quantized: true)
    embeddings = model.(sentences)

    assert_elements_in_delta [0.045927, 0.07328, 0.05401], embeddings[0][..2]
    assert_elements_in_delta [0.081881, 0.1076, -0.01324], embeddings[1][..2]
  end

  # https://huggingface.co/Xenova/multi-qa-MiniLM-L6-cos-v1
  def test_multi_qa_minilm
    query = "How many people live in London?"
    docs = ["Around 9 Million people live in London", "London is known for its financial district"]

    model = Informers.pipeline("embedding", "Xenova/multi-qa-MiniLM-L6-cos-v1")
    query_embedding = model.(query)
    doc_embeddings = model.(docs)
    scores = doc_embeddings.map { |e| e.zip(query_embedding).sum { |d, q| d * q } }
    doc_score_pairs = docs.zip(scores).sort_by { |d, s| -s }

    assert_equal "Around 9 Million people live in London", doc_score_pairs[0][0]
    assert_in_delta 0.9156, doc_score_pairs[0][1]
    assert_equal "London is known for its financial district", doc_score_pairs[1][0]
    assert_in_delta 0.4948, doc_score_pairs[1][1]
  end

  # https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
  def test_mxbai_embed
    transform_query = lambda do |query|
      "Represent this sentence for searching relevant passages: #{query}"
    end

    docs = [
      transform_query.("puppy"),
      "The dog is barking",
      "The cat is purring"
    ]

    model = Informers.pipeline("embedding", "mixedbread-ai/mxbai-embed-large-v1")
    embeddings = model.(docs, pooling: "cls", normalize: false)

    assert_elements_in_delta [-0.00624076, 0.12864432, 0.5248165], embeddings[0][..2]
    assert_elements_in_delta [-0.61227727, 1.4060247, -0.04079155], embeddings[-1][..2]
  end

  # https://huggingface.co/Supabase/gte-small
  def test_gte_small
    sentences = ["That is a happy person", "That is a very happy person"]

    model = Informers.pipeline("embedding", "Supabase/gte-small")
    embeddings = model.(sentences)

    assert_elements_in_delta [-0.05316979, 0.01044252, 0.06194701], embeddings[0][..2]
    assert_elements_in_delta [-0.05246907, 0.03752426, 0.07344585], embeddings[-1][..2]
  end

  # https://huggingface.co/intfloat/e5-base-v2
  def test_e5_base
    input = [
      "passage: Ruby is a programming language created by Matz",
      "query: Ruby creator"
    ]

    model = Informers.pipeline("embedding", "intfloat/e5-base-v2")
    embeddings = model.(input)

    assert_elements_in_delta [-0.00596662, -0.03730119, -0.0703470], embeddings[0][..2]
    assert_elements_in_delta [0.00298353, -0.04421991, -0.0591884], embeddings[-1][..2]
  end

  # https://huggingface.co/nomic-ai/nomic-embed-text-v1
  def test_nomic_embed
    input = [
      "search_document: The dog is barking",
      "search_query: puppy"
    ]

    model = Informers.pipeline("embedding", "nomic-ai/nomic-embed-text-v1")
    embeddings = model.(input)

    assert_elements_in_delta [-0.00645858, 0.01145126, 0.0099767], embeddings[0][..2]
    assert_elements_in_delta [-0.01173127, 0.04957652, -0.0176401], embeddings[-1][..2]
  end

  # https://huggingface.co/BAAI/bge-base-en-v1.5
  def test_bge_base
    transform_query = lambda do |query|
      "Represent this sentence for searching relevant passages: #{query}"
    end

    docs = [
      transform_query.("puppy"),
      "The dog is barking",
      "The cat is purring"
    ]

    model = Informers.pipeline("embedding", "BAAI/bge-base-en-v1.5")
    embeddings = model.(docs)

    assert_elements_in_delta [0.00029264, -0.0619305, -0.06199387], embeddings[0][..2]
    assert_elements_in_delta [-0.07482512, -0.0770234, 0.03398684], embeddings[-1][..2]
  end

  # https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1
  def test_mxbai_rerank
    query = "How many people live in London?"
    docs = ["Around 9 Million people live in London", "London is known for its financial district"]

    model = Informers.pipeline("reranking", "mixedbread-ai/mxbai-rerank-base-v1")
    result = model.(query, docs, return_documents: true)

    assert_equal 0, result[0][:doc_id]
    assert_in_delta 0.984, result[0][:score]
    assert_equal docs[0], result[0][:text]

    assert_equal 1, result[1][:doc_id]
    assert_in_delta 0.139, result[1][:score]
    assert_equal docs[1], result[1][:text]
  end
end