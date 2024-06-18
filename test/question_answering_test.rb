require_relative "test_helper"

class QuestionAnsweringTest < Minitest::Test
  def test_predict
    result = model.predict(question: "Who invented Ruby?", context: "Ruby is a programming language created by Matz")
    assert_answer result, "Matz", 0.9980658360049758, 42, 46
  end

  def test_predict_multiple
    questions = [
      {question: "Who invented Ruby?", context: "Ruby is a programming language created by Matz"},
      {question: "Who walked on the moon?", context: "Neil Armstrong walked on the moon."}
    ]
    result = model.predict(questions)
    assert_answer result[0], "Matz", 0.9980658360049758, 42, 46
    assert_answer result[1], "Neil Armstrong", 0.9911048638990259, 0, 14
  end

  def assert_answer(result, answer, score, start, stop)
    assert_equal answer, result[:answer]
    assert_in_delta score, result[:score], 0.01 # for Travis
    assert_equal start, result[:start]
    assert_equal stop, result[:end]
  end

  def model
    @model ||= Informers::QuestionAnswering.new("#{models_path}/question-answering.onnx")
  end
end
