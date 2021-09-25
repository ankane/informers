require_relative "test_helper"

class FillMaskTest < Minitest::Test
  def setup
    skip if ENV["CI"]
  end

  def test_predict
    result = model.predict("HuggingFace is creating a <mask> that the community uses to solve NLP tasks.")
    assert_equal "HuggingFace is creating a tool that the community uses to solve NLP tasks.", result[0][:sequence]
    assert_in_delta 0.17927613854408264, result[0][:score]
    assert_equal [3944, 7208, 5560, 8503, 17715], result.map { |r| r[:token] }
    assert_equal ["tool", "framework", "library", "database", "prototype"], result.map { |r| r[:token_str] }
  end

  def test_predict_multiple
    result = model.predict(["Hello <mask>", "Goodbye <mask>"])

    assert_equal ["Hello!", "Hello!\"", "Hello Kitty", "Hello!.", "Hello!!!"], result[0].map { |r| r[:sequence] }
    assert_in_delta 0.29753604531288147, result[0][0][:score]
    assert_equal [328, 2901, 32906, 42202, 16506], result[0].map { |r| r[:token] }
    assert_equal ["!", "!\"", "Kitty", "!.", "!!!"], result[0].map { |r| r[:token_str] }

    assert_equal ["Goodbye!", "Goodbye!\"", "Goodbye!!!", "Goodbye!!", "Goodbye :)"], result[1].map { |r| r[:sequence] }
    assert_in_delta 0.3994251787662506, result[1][0][:score]
    assert_equal [328, 2901, 16506, 12846, 44660], result[1].map { |r| r[:token] }
    assert_equal ["!", "!\"", "!!!", "!!", ":)"], result[1].map { |r| r[:token_str] }
  end

  def test_no_mask_token
    error = assert_raises do
      model.predict("Hello")
    end
    assert_equal "No mask_token (<mask>) found on the input", error.message
  end

  def test_multiple_mask_tokens
    error = assert_raises do
      model.predict("Hello <mask> <mask>")
    end
    assert_equal "More than one mask_token (<mask>) is not supported", error.message
  end

  def model
    @model ||= Informers::FillMask.new("#{models_path}/fill-mask.onnx")
  end
end
