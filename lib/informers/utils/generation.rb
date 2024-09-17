module Informers
  module Utils
    class GenerationConfig
      def initialize(kwargs)
        @config = {}

        # Parameters that control the length of the output
        @config["max_length"] = kwargs["max_length"] || 20
        @config["max_new_tokens"] = kwargs["max_new_tokens"]
        @config["min_length"] = kwargs["min_length"] || 0
        @config["min_new_tokens"] = kwargs["min_new_tokens"]
        @config["early_stopping"] = kwargs["early_stopping"] || false
        @config["max_time"] = kwargs["max_time"]

        # Parameters that control the generation strategy used
        @config["do_sample"] = kwargs["do_sample"] || false
        @config["num_beams"] = kwargs["num_beams"] || 1
        @config["num_beam_groups"] = kwargs["num_beam_groups"] || 1
        @config["penalty_alpha"] = kwargs["penalty_alpha"]
        @config["use_cache"] = kwargs.fetch("use_cache", true)

        # Parameters for manipulation of the model output logits
        @config["temperature"] = kwargs["temperature"] || 1.0
        @config["top_k"] = kwargs["top_k"] || 50
        @config["top_p"] = kwargs["top_p"] || 1.0
        @config["typical_p"] = kwargs["typical_p"] || 1.0
        @config["epsilon_cutoff"] = kwargs["epsilon_cutoff"] || 0.0
        @config["eta_cutoff"] = kwargs["eta_cutoff"] || 0.0
        @config["diversity_penalty"] = kwargs["diversity_penalty"] || 0.0
        @config["repetition_penalty"] = kwargs["repetition_penalty"] || 1.0
        @config["encoder_repetition_penalty"] = kwargs["encoder_repetition_penalty"] || 1.0
        @config["length_penalty"] = kwargs["length_penalty"] || 1.0
        @config["no_repeat_ngram_size"] = kwargs["no_repeat_ngram_size"] || 0
        @config["bad_words_ids"] = kwargs["bad_words_ids"]
        @config["force_words_ids"] = kwargs["force_words_ids"]
        @config["renormalize_logits"] = kwargs["renormalize_logits"] || false
        @config["constraints"] = kwargs["constraints"]
        @config["forced_bos_token_id"] = kwargs["forced_bos_token_id"]
        @config["forced_eos_token_id"] = kwargs["forced_eos_token_id"]
        @config["remove_invalid_values"] = kwargs["remove_invalid_values"] || false
        @config["exponential_decay_length_penalty"] = kwargs["exponential_decay_length_penalty"]
        @config["suppress_tokens"] = kwargs["suppress_tokens"]
        @config["begin_suppress_tokens"] = kwargs["begin_suppress_tokens"]
        @config["forced_decoder_ids"] = kwargs["forced_decoder_ids"]

        # Parameters that define the output variables of `generate`
        @config["num_return_sequences"] = kwargs["num_return_sequences"] || 1
        @config["output_attentions"] = kwargs["output_attentions"] || false
        @config["output_hidden_states"] = kwargs["output_hidden_states"] || false
        @config["output_scores"] = kwargs["output_scores"] || false
        @config["return_dict_in_generate"] = kwargs["return_dict_in_generate"] || false

        # Special tokens that can be used at generation time
        @config["pad_token_id"] = kwargs["pad_token_id"]
        @config["bos_token_id"] = kwargs["bos_token_id"]
        @config["eos_token_id"] = kwargs["eos_token_id"]

        # Generation parameters exclusive to encoder-decoder models
        @config["encoder_no_repeat_ngram_size"] = kwargs["encoder_no_repeat_ngram_size"] || 0
        @config["decoder_start_token_id"] = kwargs["decoder_start_token_id"]

        # Wild card
        @generation_kwargs = kwargs["generation_kwargs"] || {}
      end

      def [](key)
        @config[key.to_s]
      end

      def merge!(config)
        @config.merge!(config)
      end
    end

    class Sampler
      def initialize(generation_config)
        super()
        @generation_config = generation_config
      end

      def call(logits, index = -1)
        # Sample from logits, of dims [batch, sequence_length, vocab_size].
        # If index is specified, sample from [batch, index, vocab_size].
        sample(logits, index)
      end

      def get_logits(logits, index)
        vocab_size = Utils.dims(logits)[-1]

        logs = logits

        if index == -1
          logs = logs.map { |v| v.slice(-vocab_size) }
        else
          raise Todo
        end

        # add temperature
        if @generation_config["temperature"] > 0
          logs = logs.map { |x| x / @generation_config["temperature"] }
        end
        logs
      end

      def self.get_sampler(generation_config)
        if generation_config[:do_sample]
          MultinomialSampler.new(generation_config)
        elsif generation_config[:num_beams] > 1
          BeamSearchSampler.new(generation_config)
        else
          if generation_config[:num_return_sequences] > 1
            raise Error, "num_return_sequences has to be 1 when doing greedy search, but is #{generation_config[:num_return_sequences]}."
          end
          GreedySampler.new(generation_config)
        end
      end
    end

    class GreedySampler < Sampler
      def sample(logits, index = -1)
        # NOTE: no need to do log_softmax here since we only take the maximum
        logs = get_logits(logits, index)
        argmax = Utils.max(logs)[1]

        # Note: score is meaningless in this context, since we are performing
        # greedy search (p = 1 => log(p) = 0)
        [
          [argmax, 0]
        ]
      end
    end

    class LogitsProcessorList
      def initialize
        super
        @processors = []
      end

      def push(item)
        @processors << item
      end

      def concat(items)
        @processors.concat(items)
      end

      def call(input_ids, batched_logits)
        # NOTE: This is different from the Python code, since vanilla Ruby does not support vectorized operations.
        # As a result, we apply each processor to each item in the batch.
        batched_logits.each do |logit|
          # Modifies logits inplace
          @processors.each do |func|
            func.(input_ids, logits)
          end
        end
      end

      def to_ary
        @processors
      end
    end
  end
end
