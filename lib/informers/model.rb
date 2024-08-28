module Informers
  class Model
    def initialize(model_id, quantized: false)
      @model_id = model_id
      @model = Informers.pipeline("feature-extraction", model_id, quantized: quantized)

      # TODO better pattern
      if model_id == "sentence-transformers/all-MiniLM-L6-v2"
        @model.instance_variable_get(:@model).instance_variable_set(:@output_names, ["sentence_embedding"])
      end
    end

    def embed(texts)
      is_batched = texts.is_a?(Array)
      texts = [texts] unless is_batched

      case @model_id
      when "sentence-transformers/all-MiniLM-L6-v2"
        output = @model.(texts)
      when "Xenova/all-MiniLM-L6-v2", "Xenova/multi-qa-MiniLM-L6-cos-v1", "Supabase/gte-small", "intfloat/e5-base-v2",
           "nomic-ai/nomic-embed-text-v1"
        output = @model.(texts, pooling: "mean", normalize: true)
      when "mixedbread-ai/mxbai-embed-large-v1"
        output = @model.(texts, pooling: "cls")
      else
        raise Error, "model not supported: #{@model_id}"
      end

      is_batched ? output : output[0]
    end
  end
end
