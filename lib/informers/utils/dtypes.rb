module Informers
  module Utils
    DEFAULT_DTYPE_SUFFIX_MAPPING = {
      fp32: "",
      fp16: "_fp16",
      int8: "_int8",
      uint8: "_uint8",
      q8: "_quantized",
      q4: "_q4",
      q4f16: "_q4f16",
      bnb4: "_bnb4"
    }
  end
end
