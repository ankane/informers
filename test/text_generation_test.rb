require_relative "test_helper"

class TextGenerationTest < Minitest::Test
  def setup
    skip if ci?
  end

  def test_predict
    result = model.predict("As far as I am concerned, I will")
    assert_equal result, "As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a \"free market.\" I think that the idea of a free market is a bit of a stretch. I think that the idea"
  end

  def model
    @model ||= Informers::TextGeneration.new("#{models_path}/text-generation.onnx")
  end
end
