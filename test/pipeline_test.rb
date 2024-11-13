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

  def test_zero_shot_classification
    classifier = Informers.pipeline("zero-shot-classification")
    text = "Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app."
    labels = ["mobile", "billing", "website", "account access"]
    result = classifier.(text, labels)
    assert_equal text, result[:sequence]
    assert_equal ["mobile", "billing", "account access", "website"], result[:labels]
    assert_elements_in_delta [0.633, 0.134, 0.121, 0.111], result[:scores]
  end

  def test_text2text_generation
    text2text = Informers.pipeline("text2text-generation")
    result = text2text.("translate from English to French: I'm very happy")
    assert_equal "Je suis très heureux.", result[0][:generated_text]
  end

  def test_translation
    translator = Informers.pipeline("translation", "Xenova/nllb-200-distilled-600M")
    result = translator.("जीवन एक चॉकलेट बॉक्स की तरह है।", src_lang: "hin_Deva", tgt_lang: "fra_Latn")
    assert_equal "La vie est comme une boîte à chocolat.", result[0][:translation_text]
  end

  def test_text_generation
    generator = Informers.pipeline("text-generation")
    result = generator.("I enjoy walking with my cute dog,")
    assert_equal "I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to", result[0][:generated_text]
  end

  def test_summarization
    skip "TODO"

    summarizer = Informers.pipeline("summarization")
    result = summarizer.("Ruby is awesome.")
    assert_equal "Ruby is awesome. Ruby is awesome. Ruby is great. Ruby's website is great. Ruby's site is great for the first time. Ruby will be great for all the people who want to know more about the site. Click here for more information. Click HERE for", result[0][:summary_text]
  end

  def test_fill_mask
    unmasker = Informers.pipeline("fill-mask")
    result = unmasker.("Paris is the [MASK] of France.")
    assert_equal 5, result.size
    assert_in_delta 0.997, result[0][:score]
    assert_equal 3007, result[0][:token]
    assert_equal "capital", result[0][:token_str]
    assert_equal "paris is the capital of france.", result[0][:sequence]
  end

  def test_fill_mask_no_mask_token
    unmasker = Informers.pipeline("fill-mask")
    error = assert_raises(ArgumentError) do
      unmasker.("Paris is the <mask> of France.")
    end
    assert_equal "Mask token ([MASK]) not found in text.", error.message
  end

  def test_feature_extraction
    sentences = ["This is an example sentence", "Each sentence is converted"]
    extractor = Informers.pipeline("feature-extraction")
    output = extractor.(sentences)
    assert_in_delta (-0.0145), output[0][0][0]
    assert_in_delta (-0.3130), output[-1][-1][-1]
  end

  def test_embedding
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embed = Informers.pipeline("embedding")
    embeddings = embed.(sentences)
    assert_elements_in_delta [0.067657, 0.063496, 0.048713], embeddings[0][..2]
    assert_elements_in_delta [0.086439, 0.10276, 0.0053946], embeddings[1][..2]
  end

  def test_reranking
    query = "How many people live in London?"
    docs = ["Around 9 Million people live in London", "London is known for its financial district"]
    rerank = Informers.pipeline("reranking")
    result = rerank.(query, docs)
    assert_equal 2, result.size
    assert_equal 0, result[0][:doc_id]
    assert_in_delta 0.984, result[0][:score]
    assert_equal 1, result[1][:doc_id]
    assert_in_delta 0.139, result[1][:score]
  end

  def test_image_classification
    classifier = Informers.pipeline("image-classification")
    result = classifier.("test/support/pipeline-cat-chonk.jpeg", top_k: 2)
    assert_equal "lynx, catamount", result[0][:label]
    assert_in_delta 0.428, result[0][:score], 0.01
    assert_equal "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor", result[1][:label]
    assert_in_delta 0.047, result[1][:score], 0.01
  end

  def test_zero_shot_image_classification
    classifier = Informers.pipeline("zero-shot-image-classification")
    result = classifier.("test/support/pipeline-cat-chonk.jpeg", ["dog", "cat", "tiger"])
    assert_equal 3, result.size
    assert_equal "cat", result[0][:label]
    assert_in_delta 0.756, result[0][:score]
    assert_equal "tiger", result[1][:label]
    assert_in_delta 0.189, result[1][:score]
    assert_equal "dog", result[2][:label]
    assert_in_delta 0.055, result[2][:score]
  end

  def test_object_detection
    detector = Informers.pipeline("object-detection")
    result = detector.("test/support/pipeline-cat-chonk.jpeg")
    assert_equal 3, result.size

    assert_equal "cat", result[0][:label]
    assert_in_delta 0.992, result[0][:score]
    assert_equal 177, result[0][:box][:xmin]
    assert_equal 153, result[0][:box][:ymin]
    assert_equal 959, result[0][:box][:xmax]
    assert_equal 600, result[0][:box][:ymax]

    assert_equal "bicycle", result[2][:label]
    assert_in_delta 0.726, result[2][:score]
    assert_equal 0, result[2][:box][:xmin]
    assert_equal 0, result[2][:box][:ymin]
    assert_equal 196, result[2][:box][:xmax]
    assert_equal 413, result[2][:box][:ymax]
  end

  def test_zero_shot_object_detection
    detector = Informers.pipeline("zero-shot-object-detection")
    result = detector.("test/support/zero-sh-obj-detection_1.png", ["human face", "rocket", "helmet", "american flag"])
    assert_equal 4, result.size

    assert_equal "human face", result[0][:label]
    assert_in_delta 0.351, result[0][:score]
    assert_equal 179, result[0][:box][:xmin]
    assert_equal 72, result[0][:box][:ymin]
    assert_equal 270, result[0][:box][:xmax]
    assert_equal 178, result[0][:box][:ymax]

    assert_equal "rocket", result[1][:label]
    assert_in_delta 0.211, result[1][:score]
    assert_equal 351, result[1][:box][:xmin]
    assert_equal 6, result[1][:box][:ymin]
    assert_equal 468, result[1][:box][:xmax]
    assert_equal 289, result[1][:box][:ymax]
  end

  def test_depth_estimation
    estimator = Informers.pipeline("depth-estimation")
    result = estimator.("test/support/pipeline-cat-chonk.jpeg")
    assert_in_delta 1.078, result[:predicted_depth][0][0]
    assert_kind_of Vips::Image, result[:depth]
    # result[:depth].write_to_file("/tmp/depth-estimation.jpg")
  end

  def test_image_to_text
    captioner = Informers.pipeline("image-to-text")
    result = captioner.("test/support/pipeline-cat-chonk.jpeg")
    assert_equal "a cat is standing in the snow", result[0][:generated_text]
  end

  def test_image_to_image
    skip "Expensive"

    upscaler = Informers.pipeline("image-to-image")
    result = upscaler.("test/support/pipeline-cat-chonk.jpeg")
    assert_kind_of Vips::Image, result
    result.write_to_file("/tmp/image-to-image.jpg")
  end

  def test_image_segmentation
    segmenter = Informers.pipeline("image-segmentation")
    result = segmenter.("test/support/pipeline-cat-chonk.jpeg")
    assert_equal 3, result.size

    assert_equal "snow", result[0][:label]
    assert_in_delta 0.997, result[0][:score]
    assert_equal "LABEL_184", result[1][:label]
    assert_in_delta 0.993, result[1][:score]
    assert_equal "cat", result[2][:label]
    assert_in_delta 0.998, result[2][:score]
  end

  def test_image_feature_extraction
    fe = Informers.pipeline("image-feature-extraction")
    result = fe.("test/support/pipeline-cat-chonk.jpeg")
    assert_in_delta 0.877, result[0][0], 0.01
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

  def test_device
    skip unless mac?

    sentences = ["This is an example sentence", "Each sentence is converted"]
    embed = Informers.pipeline("embedding", "Xenova/all-MiniLM-L6-v2", device: "coreml")
    embeddings = embed.(sentences)
    assert_elements_in_delta [0.067657, 0.063496, 0.048713], embeddings[0][..2]
    assert_elements_in_delta [0.086439, 0.10276, 0.0053946], embeddings[1][..2]
  end

  def test_dtype
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embed = Informers.pipeline("embedding", "Xenova/all-MiniLM-L6-v2", dtype: "fp16")
    embeddings = embed.(sentences)
    assert_elements_in_delta [0.067657, 0.063496, 0.048713], embeddings[0][..2]
    assert_elements_in_delta [0.086439, 0.10276, 0.0053946], embeddings[1][..2]
  end
end
