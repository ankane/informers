module Informers
  class Model
    def initialize(model_id, quantized: false)
      @model = Informers.pipeline("embedding", model_id, quantized: quantized)
      @options = model_id == "mixedbread-ai/mxbai-embed-large-v1" ? {pooling: "cls", normalize: false} : {}
    end

    def embed(texts)
      @model.(texts, **@options)
    end
  end
end
