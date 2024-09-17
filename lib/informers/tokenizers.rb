module Informers
  class PreTrainedTokenizer
    attr_reader :mask_token, :mask_token_id, :sep_token_id

    def initialize(tokenizer_json, tokenizer_config)
      super()

      @tokenizer = Tokenizers::Tokenizer.from_file(tokenizer_json)

      @mask_token = tokenizer_config["mask_token"]
      @mask_token_id = @tokenizer.token_to_id(@mask_token) if @mask_token

      @sep_token = tokenizer_config["sep_token"]
      @sep_token_id = @tokenizer.token_to_id(@sep_token) if @sep_token

      @model_max_length = tokenizer_config["model_max_length"]
    end

    def call(
      text,
      text_pair: nil,
      add_special_tokens: true,
      padding: false,
      truncation: nil,
      max_length: nil,
      return_tensor: true,
      return_token_type_ids: true, # TODO change default
      return_offsets: false
    )
      is_batched = text.is_a?(Array)

      if is_batched
        if text.length == 0
          raise Error, "text array must be non-empty"
        end

        if !text_pair.nil?
          if !text_pair.is_a?(Array)
            raise Error, "text_pair must also be an array"
          elsif text.length != text_pair.length
            raise Error, "text and text_pair must have the same length"
          end
        end
      end

      if padding
        @tokenizer.enable_padding
      else
        @tokenizer.no_padding
      end

      if truncation
        @tokenizer.enable_truncation(max_length || @model_max_length)
      else
        @tokenizer.no_truncation
      end

      if is_batched
        input = text_pair ? text.zip(text_pair) : text
        encoded = @tokenizer.encode_batch(input, add_special_tokens: add_special_tokens)
      else
        encoded = [@tokenizer.encode(text, text_pair, add_special_tokens: add_special_tokens)]
      end

      result = {input_ids: encoded.map(&:ids), attention_mask: encoded.map(&:attention_mask)}
      if return_token_type_ids
        result[:token_type_ids] = encoded.map(&:type_ids)
      end
      if return_offsets
        result[:offsets] = encoded.map(&:offsets)
      end
      result
    end

    def decode(tokens, skip_special_tokens:)
      @tokenizer.decode(tokens, skip_special_tokens: skip_special_tokens)
    end

    def convert_tokens_to_string(tokens)
      @tokenizer.decoder.decode(tokens)
    end

    def id_to_token(id)
      @tokenizer.id_to_token(id)
    end
  end

  class BertTokenizer < PreTrainedTokenizer
    # TODO
    # self.return_token_type_ids = true
  end

  class DebertaV2Tokenizer < PreTrainedTokenizer
    # TODO
    # self.return_token_type_ids = true
  end

  class DistilBertTokenizer < PreTrainedTokenizer
  end

  class T5Tokenizer < PreTrainedTokenizer
  end

  class GPT2Tokenizer < PreTrainedTokenizer
    # _default_chat_template = `{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}`
  end

  class BartTokenizer < PreTrainedTokenizer
  end

  class RobertaTokenizer < PreTrainedTokenizer
  end

  class XLMRobertaTokenizer < PreTrainedTokenizer
  end

  class MPNetTokenizer < PreTrainedTokenizer
  end

  class CLIPTokenizer < PreTrainedTokenizer
  end

  class AutoTokenizer
    TOKENIZER_CLASS_MAPPING = {
      "T5Tokenizer" => T5Tokenizer,
      "BertTokenizer" => BertTokenizer,
      "DebertaV2Tokenizer" => DebertaV2Tokenizer,
      "DistilBertTokenizer" => DistilBertTokenizer,
      "BartTokenizer" => BartTokenizer,
      "RobertaTokenizer" => RobertaTokenizer,
      "XLMRobertaTokenizer" => XLMRobertaTokenizer,
      "MPNetTokenizer" => MPNetTokenizer,
      "CLIPTokenizer" => CLIPTokenizer
    }

    def self.from_pretrained(
      pretrained_model_name_or_path,
      quantized: true,
      progress_callback: nil,
      config: nil,
      cache_dir: nil,
      local_files_only: false,
      revision: "main",
      legacy: nil,
      **kwargs
    )
      tokenizer_json, tokenizer_config = load_tokenizer(
        pretrained_model_name_or_path,
        quantized:,
        progress_callback:,
        config:,
        cache_dir:,
        local_files_only:,
        revision:,
        legacy:
      )

      # Some tokenizers are saved with the "Fast" suffix, so we remove that if present.
      tokenizer_name = tokenizer_config["tokenizer_class"]&.delete_suffix("Fast") || "PreTrainedTokenizer"

      cls = TOKENIZER_CLASS_MAPPING[tokenizer_name]
      if !cls
        warn "Unknown tokenizer class #{tokenizer_name.inspect}, attempting to construct from base class."
        cls = PreTrainedTokenizer
      end
      cls.new(tokenizer_json, tokenizer_config)
    end

    def self.load_tokenizer(pretrained_model_name_or_path, **options)
      info = [
        Utils::Hub.get_model_file(pretrained_model_name_or_path, "tokenizer.json", true, **options),
        Utils::Hub.get_model_json(pretrained_model_name_or_path, "tokenizer_config.json", true, **options),
      ]

      # Override legacy option if `options.legacy` is not null
      if !options[:legacy].nil?
        info[1]["legacy"] = options[:legacy]
      end
      info
    end
  end
end
