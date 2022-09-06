require_relative "test_helper"

class NERTest < Minitest::Test
  def setup
    skip if ci?
  end

  def test_predict
    entities = model.predict("Nat works at GitHub in San Francisco")
    assert_equal 3, entities.size
    assert_entity entities[0], "Nat", "person", 0.9878539543687016, 0, 3
    assert_entity entities[1], "GitHub", "org", 0.9255155490518876, 13, 19
    assert_entity entities[2], "San Francisco", "location", 0.9976330361603447, 23, 36
  end

  def test_predict_multiple
    result = model.predict(["Nat works at GitHub in San Francisco", "I love New York"])
    assert_equal 2, result.size

    assert_equal 3, result[0].size
    assert_entity result[0][0], "Nat", "person", 0.9878539543687016, 0, 3
    assert_entity result[0][1], "GitHub", "org", 0.9255155490518876, 13, 19
    assert_entity result[0][2], "San Francisco", "location", 0.9976330361603447, 23, 36

    assert_equal 1, result[1].size
    # .96 -> .86 when quantized
    assert_entity result[1][0], "New York", "location", 0.8598015581299177, 7, 15, 0.11
  end

  def assert_entity(entity, text, tag, score, start, stop, delta = 0.03)
    assert_equal text, entity[:text]
    assert_equal tag, entity[:tag]
    assert_in_delta score, entity[:score], delta
    assert_equal start, entity[:start]
    assert_equal stop, entity[:end]
  end

  def model
    @model ||= Informers::NER.new("#{models_path}/ner.onnx")
  end
end
