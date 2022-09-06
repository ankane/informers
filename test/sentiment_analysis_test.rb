require_relative "test_helper"

class SentimentAnalysisTest < Minitest::Test
  def test_predict
    result = model.predict("This is super cool")
    assert_sentiment result, "positive", 0.999855186578301
  end

  def test_predict_multiple
    result = model.predict(["This is super cool", "I didn't like it"])
    assert_sentiment result[0], "positive", 0.9998597058884394
    assert_sentiment result[1], "negative",  0.9990689893891818
  end

  def assert_sentiment(result, label, score)
    assert_equal label, result[:label]
    assert_in_delta score, result[:score], 0.01
  end

  def model
    @model ||= Informers::SentimentAnalysis.new("#{models_path}/sentiment-analysis.onnx")
  end
end
