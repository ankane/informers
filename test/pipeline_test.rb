require_relative "test_helper"

class PipelineTest < Minitest::Test
  def test_ner
    ner = Informers.pipeline("ner")
    result = ner.("Ruby is a programming language created by Matz")
    assert_equal 1, result.size
    assert_equal "PER", result[0][:entity_group]
    assert_in_delta 0.994, result[0][:score]
    assert_equal "Matz", result[0][:word]
    assert_equal 42, result[0][:start]
    assert_equal 46, result[0][:end]
  end

  def test_ner_aggregation_strategy
    ner = Informers.pipeline("ner")
    result = ner.("Ruby is a programming language created by Matz", aggregation_strategy: "none")
    assert_equal 2, result.size
    assert_equal "B-PER", result[0][:entity]
    assert_in_delta 0.996, result[0][:score]
    assert_equal 8, result[0][:index]
    assert_equal "Mat", result[0][:word]
    assert_equal 42, result[0][:start]
    assert_equal 45, result[0][:end]
  end

  def test_sentiment_analysis
    classifier = Informers.pipeline("sentiment-analysis")
    result = classifier.("I love transformers!")
    assert_equal "POSITIVE", result[:label]
    assert_in_delta 0.9997887, result[:score], 0.0000001

    result = classifier.("This is super cool")
    assert_equal "POSITIVE", result[:label]
    assert_in_delta 0.9998608, result[:score], 0.0000001

    result = classifier.(["This is super cool", "I didn't like it"])
    assert_equal "POSITIVE", result[0][:label]
    assert_in_delta 0.9998600, result[0][:score], 0.0000001
    assert_equal "NEGATIVE", result[1][:label]
    assert_in_delta 0.9985375, result[1][:score], 0.0000001
  end

  def test_question_answering
    qa = Informers.pipeline("question-answering")
    result = qa.("Who invented Ruby?", "Ruby is a programming language created by Matz")
    assert_in_delta 0.998, result[:score]
    assert_equal "Matz", result[:answer]
    assert_equal 42, result[:start]
    assert_equal 46, result[:end]
  end

  def test_feature_extraction
    sentences = ["This is an example sentence", "Each sentence is converted"]
    extractor = Informers.pipeline("feature-extraction")
    output = extractor.(sentences)
    assert_in_delta (-0.0145), output[0][0][0]
    assert_in_delta (-0.3130), output[-1][-1][-1]
  end

  def test_rerank
    ranker = Informers.pipeline("rerank")
    result = ranker.("Who created Ruby?", ["Matz created Ruby", "Another doc"])
    assert_equal 2, result.size
    assert_equal 0, result[0][:doc_id]
    assert_equal 1, result[1][:doc_id]
  end

  def test_progress_callback
    msgs = []
    extractor = Informers.pipeline("feature-extraction", progress_callback: ->(msg) { msgs << msg })
    extractor.("I love transformers!")

    expected_msgs = [
      {status: "initiate", name: "Xenova/all-MiniLM-L6-v2", file: "tokenizer.json"},
      {status: "ready", task: "feature-extraction", model: "Xenova/all-MiniLM-L6-v2"}
    ]
    expected_msgs.each do |expected|
      assert_includes msgs, expected
    end
  end
end
