require_relative "test_helper"

class FeatureExtractionTest < Minitest::Test
  def test_predict
    result = model.predict("This is a test")
    assert_equal 6, result.size
    assert_equal 768, result.first.size
  end

  def test_predict_multiple
    result = model.predict(["This is a test", "Another test"])
    assert_equal 2, result.size
    assert_equal 6, result.first.size
    assert_equal 768, result.first.first.size
  end

  def model
    @model ||= Informers::FeatureExtraction.new("#{models_path}/feature-extraction.onnx")
  end
end
