require_relative "test_helper"

class FeatureExtractionTest < Minitest::Test
  def setup
    skip if ENV["CI"]
  end

  def test_predict
    result = model.predict("This is a test")
    # TODO improve
    assert result
  end

  def test_predict_multiple
    result = model.predict(["This is a test", "Another test"])
    # TODO improve
    assert result
  end

  def model
    @model ||= Informers::FeatureExtraction.new("#{models_path}/feature-extraction.onnx")
  end
end
