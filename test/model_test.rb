require_relative "test_helper"

class ModelTest < Minitest::Test
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
    query_prefix = "Represent this sentence for searching relevant passages: "

    input = [
      "The dog is barking",
      "The cat is purring",
      query_prefix + "puppy"
    ]

    model = Informers.pipeline("embedding", "mixedbread-ai/mxbai-embed-large-v1")
    embeddings = model.(input, pooling: "cls", normalize: false)

    assert_elements_in_delta [-0.61227727, 1.4060247, -0.04079155], embeddings[1][..2]
    assert_elements_in_delta [-0.00624076, 0.12864432, 0.5248165], embeddings[-1][..2]
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
    doc_prefix = "passage: "
    query_prefix = "query: "

    input = [
      doc_prefix + "Ruby is a programming language created by Matz",
      query_prefix + "Ruby creator"
    ]

    model = Informers.pipeline("embedding", "intfloat/e5-base-v2")
    embeddings = model.(input)

    assert_elements_in_delta [-0.00596662, -0.03730119, -0.0703470], embeddings[0][..2]
    assert_elements_in_delta [0.00298353, -0.04421991, -0.0591884], embeddings[-1][..2]
  end

  # https://huggingface.co/nomic-ai/nomic-embed-text-v1
  def test_nomic_embed
    doc_prefix = "search_document: "
    query_prefix = "search_query: "

    input = [
      doc_prefix + "The dog is barking",
      query_prefix + "puppy"
    ]

    model = Informers.pipeline("embedding", "nomic-ai/nomic-embed-text-v1")
    embeddings = model.(input)

    assert_elements_in_delta [-0.00645858, 0.01145126, 0.0099767], embeddings[0][..2]
    assert_elements_in_delta [-0.01173127, 0.04957652, -0.0176401], embeddings[-1][..2]
  end

  # https://huggingface.co/BAAI/bge-base-en-v1.5
  def test_bge_base
    query_prefix = "Represent this sentence for searching relevant passages: "

    input = [
      "The dog is barking",
      "The cat is purring",
      query_prefix + "puppy"
    ]

    model = Informers.pipeline("embedding", "BAAI/bge-base-en-v1.5")
    embeddings = model.(input)

    assert_elements_in_delta [-0.07482512, -0.0770234, 0.03398684], embeddings[1][..2]
    assert_elements_in_delta [0.00029264, -0.0619305, -0.06199387], embeddings[-1][..2]
  end

  # https://huggingface.co/jinaai/jina-embeddings-v2-base-en
  def test_jina_embeddings
    sentences = ["How is the weather today?", "What is the current weather like today?"]

    model = Informers.pipeline("embedding", "jinaai/jina-embeddings-v2-base-en", model_file_name: "../model")
    embeddings = model.(sentences)

    assert_elements_in_delta [-0.02488641, -0.0429398, 0.04303398], embeddings[0][..2]
    assert_elements_in_delta [-0.0081194, -0.06225249, 0.03116853], embeddings[1][..2]
  end

  # https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5
  def test_snowflake_arctic_embed
    query_prefix = "Represent this sentence for searching relevant passages: "

    input = [
      "The dog is barking",
      "The cat is purring",
      query_prefix + "puppy"
    ]

    model = Informers.pipeline("embedding", "Snowflake/snowflake-arctic-embed-m-v1.5")
    embeddings = model.(input, pooling: "cls", model_output: "token_embeddings")

    assert_elements_in_delta [0.03239886, 0.0009998, 0.08401278], embeddings[0][..2]
    assert_elements_in_delta [-0.02530634, -0.02715422, 0.01218867], embeddings[-1][..2]

    embeddings = model.(input, pooling: "none", normalize: false, model_output: "sentence_embedding")

    assert_elements_in_delta [0.03239886, 0.0009998, 0.08401278], embeddings[0][..2]
    assert_elements_in_delta [-0.02530634, -0.02715422, 0.01218867], embeddings[-1][..2]
  end

  # https://huggingface.co/Xenova/all-mpnet-base-v2
  def test_all_mpnet
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = Informers.pipeline("embedding", "Xenova/all-mpnet-base-v2")
    embeddings = model.(sentences)

    assert_elements_in_delta [0.02250263, -0.07829167, -0.02303071], embeddings[0][..2]
    assert_elements_in_delta [0.04170236, 0.00109747, -0.01553415], embeddings[1][..2]
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

  # https://huggingface.co/jinaai/jina-reranker-v1-turbo-en
  def test_jina_reranker
    query = "How many people live in London?"
    docs = ["Around 9 Million people live in London", "London is known for its financial district"]

    model = Informers.pipeline("reranking", "jinaai/jina-reranker-v1-turbo-en")
    result = model.(query, docs, return_documents: true)

    assert_equal 0, result[0][:doc_id]
    assert_in_delta 0.912, result[0][:score]
    assert_equal docs[0], result[0][:text]

    assert_equal 1, result[1][:doc_id]
    assert_in_delta 0.0555, result[1][:score]
    assert_equal docs[1], result[1][:text]
  end

  # https://huggingface.co/BAAI/bge-reranker-base
  def test_bge_reranker
    query = "How many people live in London?"
    docs = ["Around 9 Million people live in London", "London is known for its financial district"]

    model = Informers.pipeline("reranking", "BAAI/bge-reranker-base")
    result = model.(query, docs, return_documents: true)

    assert_equal 0, result[0][:doc_id]
    assert_in_delta 0.996, result[0][:score]
    assert_equal docs[0], result[0][:text]

    assert_equal 1, result[1][:doc_id]
    assert_in_delta 0.000158, result[1][:score], 0.000001
    assert_equal docs[1], result[1][:text]
  end
end
