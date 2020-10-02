module Informers
  class SentimentAnalysis
    def initialize(model_path)
      tokenizer_path = File.expand_path("../../vendor/bert_base_tok.bin", __dir__)
      @tokenizer = BlingFire.load_model(tokenizer_path)
      @model = OnnxRuntime::Model.new(model_path)
    end

    def predict(texts)
      singular = !texts.is_a?(Array)
      texts = [texts] if singular

      # tokenize
      input_ids =
        texts.map do |text|
          tokens = @tokenizer.text_to_ids(text, nil, 100) # unk token
          tokens.unshift(101) # cls token
          tokens << 102 # sep token
          tokens
        end

      max_tokens = input_ids.map(&:size).max
      attention_mask = []
      input_ids.each do |ids|
        zeros = [0] * (max_tokens - ids.size)

        mask = ([1] * ids.size) + zeros
        attention_mask << mask

        ids.concat(zeros)
      end

      # infer
      input = {
        input_ids: input_ids,
        attention_mask: attention_mask
      }
      output = @model.predict(input)

      # transform
      scores =
        output["output_0"].map do |row|
          mapped = row.map { |v| Math.exp(v) }
          sum = mapped.sum
          mapped.map { |v| v / sum }
        end

      labels = ["negative", "positive"]
      scores.map! do |item|
        {label: labels[item.each_with_index.max[1]], score: item.max}
      end

      singular ? scores.first : scores
    end
  end
end
