module Informers
  class PreTrainedTokenizer
    attr_reader :mask_token, :mask_token_id, :sep_token_id

    def initialize(tokenizer_json, tokenizer_config)
      super()

      @tokenizer_config = tokenizer_config

      @tokenizer = Tokenizers::Tokenizer.from_file(tokenizer_json)

      # Add added_tokens to model
      @special_tokens = []
      @all_special_ids = []

      @added_tokens = []
      @tokenizer.added_tokens_decoder.each do |id, token|
        @added_tokens << token

        if token.special
          @special_tokens << token.content
          @all_special_ids << id
        end
      end

      # Update additional_special_tokens
      @additional_special_tokens = tokenizer_config["additional_special_tokens"] || []
      @special_tokens.concat(@additional_special_tokens)

      @mask_token = get_token("mask_token")
      @mask_token_id = @tokenizer.token_to_id(@mask_token) if @mask_token

      @sep_token = get_token("sep_token")
      @sep_token_id = @tokenizer.token_to_id(@sep_token) if @sep_token

      @model_max_length = tokenizer_config["model_max_length"]

      # for donut-base-finetuned-docvqa
      if @model_max_length && @model_max_length > (1 << 63)
        @model_max_length = 1 << 63
      end
    end

    def get_token(*keys)
      keys.each do |key|
        item = @tokenizer_config[key]
        if !item
          next
        end

        if item.is_a?(Hash)
          if item["__type"] == "AddedToken"
            return item["content"]
          else
            raise Error, "Unknown token: #{item}"
          end
        else
          return item
        end
      end

      nil
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

    def convert_tokens_to_ids(tokens)
      tokens.map { |t| @tokenizer.token_to_id(t) }
    end

    def id_to_token(id)
      @tokenizer.id_to_token(id)
    end

    def batch_decode(batch, **decode_args)
      @tokenizer.decode_batch(batch, **decode_args)
    end

    def padding_side=(side)
      @tokenizer.enable_padding(direction: side)
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

  class NllbTokenizer < PreTrainedTokenizer
    attr_reader :language_regex, :language_codes, :lang_to_token

    def initialize(tokenizer_json, tokenizer_config)
      super(tokenizer_json, tokenizer_config)

      @language_regex = /^[a-z]{3}_[A-Z][a-z]{3}$/
      @language_codes = @special_tokens.filter { |x| @language_regex.match?(x) }
      @lang_to_token = ->(x) { x } # Identity function
    end

    def _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs)
      Utils._build_translation_inputs(self, raw_inputs, tokenizer_options, generate_kwargs)
    end
  end

  class M2M100Tokenizer < PreTrainedTokenizer
    attr_reader :language_regex, :language_codes, :lang_to_token

    def initialize(tokenizer_json, tokenizer_config)
      super(tokenizer_json, tokenizer_config)

      @language_regex = /^__[a-z]{2,3}__$/
      @language_codes = @special_tokens
        .filter { |x| @language_regex.match?(x) }
        .map { |x| x.slice(2, -2) }
      @lang_to_token = ->(x) { "__#{x}__" }
    end

    def _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs)
      Utils._build_translation_inputs(self, raw_inputs, tokenizer_options, generate_kwargs)
    end
  end

  module Utils
    def self._build_translation_inputs(slf, raw_inputs, tokenizer_options, generate_kwargs)
      if !slf.respond_to?(:language_codes) || !slf.language_codes.is_a?(Array)
        raise Error, "Tokenizer must have `language_codes` attribute set and it should be an array of language ids."
      end
      if !slf.respond_to?(:language_regex) || !slf.language_regex.is_a?(Regexp)
        raise Error, "Tokenizer must have `language_regex` attribute set and it should be a regular expression."
      end
      if !slf.respond_to?(:lang_to_token) || !slf.lang_to_token.respond_to?(:call)
        raise Error, "Tokenizer must have `lang_to_token` attribute set and it should be a function."
      end
      src_lang_token = generate_kwargs[:src_lang]
      tgt_lang_token = generate_kwargs[:tgt_lang]

      if !slf.language_codes.include?(tgt_lang_token)
        raise Error, "Target language code #{tgt_lang_token.inspect} is not valid. Must be one of: #{slf.language_codes.join(", ")}"
      end

      if !src_lang_token.nil?
        # Check that the source language is valid:
        if !slf.language_codes.include?(src_lang_token)
          raise Error, "Source language code #{src_lang_token.inspect} is not valid. Must be one of: #{slf.language_codes.join(", ")}"
        end
      end

      # Override the `forced_bos_token_id` to force the correct language
      generate_kwargs["forced_bos_token_id"] = slf.convert_tokens_to_ids([slf.lang_to_token.(tgt_lang_token)])[0]

      slf.(raw_inputs, **tokenizer_options)
    end
  end

  class SpeechT5Tokenizer < PreTrainedTokenizer
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
      "CLIPTokenizer" => CLIPTokenizer,
      "GPT2Tokenizer" => GPT2Tokenizer,
      "NllbTokenizer" => NllbTokenizer,
      "M2M100Tokenizer" => M2M100Tokenizer,
      "SpeechT5Tokenizer" => SpeechT5Tokenizer
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
