module Informers
  class Model
    def initialize(model_id, quantized: false)
      @model_id = model_id
      @model = Informers.pipeline("embedding", model_id, quantized: quantized)
    end

    def embed(texts)
      case @model_id
      when "sentence-transformers/all-MiniLM-L6-v2", "Xenova/all-MiniLM-L6-v2", "Xenova/multi-qa-MiniLM-L6-cos-v1", "Supabase/gte-small"
        @model.(texts)
      when "mixedbread-ai/mxbai-embed-large-v1"
        @model.(texts, pooling: "cls", normalize: false)
      else
        raise Error, "Use the embedding pipeline for this model: #{@model_id}"
      end
    end
  end
end
