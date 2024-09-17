module Informers
  MODEL_TYPES = {
    EncoderOnly: 0,
    EncoderDecoder: 1,
    Seq2Seq: 2,
    Vision2Seq: 3,
    DecoderOnly: 4,
    MaskGeneration: 5
  }

  # NOTE: These will be populated fully later
  MODEL_TYPE_MAPPING = {}
  MODEL_NAME_TO_CLASS_MAPPING = {}
  MODEL_CLASS_TO_NAME_MAPPING = {}

  class PretrainedMixin
    def self.from_pretrained(
      pretrained_model_name_or_path,
      quantized: true,
      progress_callback: nil,
      config: nil,
      cache_dir: nil,
      local_files_only: false,
      revision: "main",
      model_file_name: nil
    )
      options = {
        quantized:,
        progress_callback:,
        config:,
        cache_dir:,
        local_files_only:,
        revision:,
        model_file_name:
      }
      config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **options)
      if options[:config].nil?
        # If no config was passed, reuse this config for future processing
        options[:config] = config
      end

      if !const_defined?(:MODEL_CLASS_MAPPINGS)
        raise Error, "`MODEL_CLASS_MAPPINGS` not implemented for this type of `AutoClass`: #{name}"
      end

      const_get(:MODEL_CLASS_MAPPINGS).each do |model_class_mapping|
        model_info = model_class_mapping[config[:model_type]]
        if !model_info
          next # Item not found in this mapping
        end
        return model_info[1].from_pretrained(pretrained_model_name_or_path, **options)
      end

      if const_defined?(:BASE_IF_FAIL)
        warn "Unknown model class #{config[:model_type].inspect}, attempting to construct from base class."
        PreTrainedModel.from_pretrained(pretrained_model_name_or_path, **options)
      else
        raise Error, "Unsupported model type: #{config[:model_type]}"
      end
    end
  end

  class PreTrainedModel
    attr_reader :config

    def initialize(config, session)
      super()

      @config = config
      @session = session

      @output_names = nil

      model_name = MODEL_CLASS_TO_NAME_MAPPING[self.class]
      model_type = MODEL_TYPE_MAPPING[model_name]

      case model_type
      when MODEL_TYPES[:DecoderOnly]
        @can_generate = true

        @run_beam = method(:decoder_run_beam)
        @get_start_beams = method(:decoder_start_beams)
        @update_beam = method(:decoder_update_beam)
        @forward = method(:decoder_forward)
      when MODEL_TYPES[:Seq2Seq], MODEL_TYPES[:Vision2Seq]
        @can_generate = true

        @run_beam = method(:seq2seq_run_beam)
        @get_start_beams = method(:seq2seq_start_beams)
        @update_beam = method(:seq2seq_update_beam)
        @forward = method(:seq2seq_forward)
      when MODEL_TYPES[:EncoderDecoder]
        raise Todo
      else
        @forward = method(:encoder_forward)
      end
    end

    def self.from_pretrained(
      pretrained_model_name_or_path,
      quantized: true,
      progress_callback: nil,
      config: nil,
      cache_dir: nil,
      local_files_only: false,
      revision: "main",
      model_file_name: nil
    )
      options = {
        quantized:,
        progress_callback:,
        config:,
        cache_dir:,
        local_files_only:,
        revision:,
        model_file_name:
      }

      model_name = MODEL_CLASS_TO_NAME_MAPPING[self]
      model_type = MODEL_TYPE_MAPPING[model_name]

      if model_type == MODEL_TYPES[:DecoderOnly]
        raise Todo

      elsif model_type == MODEL_TYPES[:Seq2Seq] || model_type == MODEL_TYPES[:Vision2Seq]
        info = [
          AutoConfig.from_pretrained(pretrained_model_name_or_path, **options),
          construct_session(pretrained_model_name_or_path, "encoder_model", **options),
          construct_session(pretrained_model_name_or_path, "decoder_model_merged", **options),
          Utils::Hub.get_model_json(pretrained_model_name_or_path, "generation_config.json", false, **options)
        ]

      elsif model_type == MODEL_TYPES[:MaskGeneration]
        raise Todo

      elsif model_type == MODEL_TYPES[:EncoderDecoder]
        raise Todo

      else
        if model_type != MODEL_TYPES[:EncoderOnly]
          warn "Model type for '#{model_name || config[:model_type]}' not found, assuming encoder-only architecture. Please report this."
        end
        info = [
          AutoConfig.from_pretrained(pretrained_model_name_or_path, **options),
          construct_session(pretrained_model_name_or_path, options[:model_file_name] || "model", **options)
        ]
      end

      new(*info)
    end

    def self.construct_session(pretrained_model_name_or_path, file_name, **options)
      prefix = "onnx/"
      if file_name.start_with?("../")
        prefix = ""
        file_name = file_name[3..]
      elsif file_name.start_with?("/")
        prefix = ""
        file_name = file_name[1..]
      end
      model_file_name = "#{prefix}#{file_name}#{options[:quantized] ? "_quantized" : ""}.onnx"
      path = Utils::Hub.get_model_file(pretrained_model_name_or_path, model_file_name, true, **options)

      OnnxRuntime::InferenceSession.new(path)
    end

    def call(model_inputs, **kwargs)
      @forward.(model_inputs, **kwargs)
    end

    def generate(inputs, generation_config = nil, logits_processor = nil, inputs_attention_mask: nil)
      if !@can_generate
        model_name = MODEL_CLASS_TO_NAME_MAPPING[self.class]
        error_message = "The current model class (#{model_name}) is not compatible with `.generate()`, as it doesn't have a language model head."
        raise Error, error_message
      end

      if !inputs.is_a?(Array)
        raise ArgumentError, "`inputs` must be an Array, but is #{inputs.class.name}"
      end

      if @config[:is_encoder_decoder]
        # Generating from the encoder outputs
        input_ids_seq_length = 0
      else
        raise Todo
      end

      # Update generation config with defaults
      generation_config = get_generation_config(generation_config)

      logits_processor ||= Utils::LogitsProcessorList.new

      # Update logits processor
      logits_processor = get_logits_processor(
        generation_config,
        input_ids_seq_length,
        logits_processor
      )

      eos_token_ids = generation_config[:eos_token_id]
      if !eos_token_ids.nil? && !eos_token_ids.is_a?(Array)
        eos_token_ids = [eos_token_ids]
      end

      num_output_tokens = 1
      max_output_tokens = num_output_tokens + (generation_config[:max_new_tokens] || Float::INFINITY)

      # Only use max length if max_new_tokens is not provided
      use_max_length = generation_config[:max_length].is_a?(Integer) && generation_config[:max_new_tokens].nil?
      sampler = Utils::Sampler.get_sampler(generation_config)

      beams = get_start_beams(inputs, generation_config, num_output_tokens, inputs_attention_mask)

      while beams.any? { |x| !x[:done] } && num_output_tokens < max_output_tokens
        newest_beams = []
        beams.each do |beam|
          if beam[:done]
            # Add this beam back into the pool
            newest_beams << beam
            next
          end
          if use_max_length && beam[:output_token_ids].length >= generation_config["max_length"]
            # Set this beam to done and add it back into the pool
            beam[:done] = true
            newest_beams << beam
            next
          end

          output = run_beam(beam)

          # add attentions/scores to beam only if user requested
          if generation_config["output_attentions"]
            add_attentions_to_beam(beam, output)
          end

          # Logits are of the form [batch_size, out_seq_length, vocab_size]
          # In most cases, this will be [batch_size, 1, vocab_size]
          # So, we select the last token's logits:
          # (equivalent to `logits = outputs.logits[:, -1, :]`)
          logits =
            output["logits"].map do |v|
              v.map { |v2| v2[-1] }
            end

          # Apply logits processor
          logits_processor.(beam[:output_token_ids], logits)

          sampled_tokens = sampler.(logits)
          sampled_tokens.each do |new_token_id, log_prob|
            # use previous beam as a starting point
            new_beam = beam.dup

            # update new beam
            update_beam(new_beam, new_token_id)

            new_beam[:score] += log_prob

            if eos_token_ids && eos_token_ids.include?(new_token_id)
              new_beam[:done] = true
            end

            newest_beams << new_beam
          end
        end
        num_output_tokens += 1

        # Next, we get the best beams, per ID
        newest_beams =
          group_beams(newest_beams).map do |group|
            group.sort_by { |v| -v[:score] }[0...generation_config["num_beams"]]
          end

        # Flatten beams
        beams = newest_beams.flatten(1)

        # Run callback
        if generation_config["callback_function"]
          generation_config["callback_function"].(beams)
        end
      end

      # TODO: Ensure that we can return non-batched outputs

      grouped_beams = group_beams(beams)

      get_flattened = lambda do |key|
        grouped_beams.map do |batch|
          if generation_config["num_return_sequences"] > 1
            raise Todo
          else
            [batch[0][key]]
          end
        end.flatten(1)
      end

      sequences = get_flattened.(:output_token_ids) # [1, seqLength]

      if generation_config["return_dict_in_generate"]
        raise Todo
      else
        sequences
      end
    end

    private

    def get_logits_processor(
      generation_config,
      input_ids_seq_length,
      logits_processor = nil
    )
      processors = Utils::LogitsProcessorList.new

      if !generation_config["repetition_penalty"].nil? && generation_config["repetition_penalty"] != 1.0
        processors.push(RepetitionPenaltyLogitsProcessor.new(generation_config["repetition_penalty"]))
      end

      if !generation_config["no_repeat_ngram_size"].nil? && generation_config["no_repeat_ngram_size"] > 0
        processors.push(NoRepeatNGramLogitsProcessor.new(generation_config["no_repeat_ngram_size"]))
      end

      if !generation_config["bad_words_ids"].nil?
        processors.push(NoBadWordsLogitsProcessor.new(generation_config["bad_words_ids"], generation_config["eos_token_id"]))
      end

      if !generation_config["min_length"].nil? && !generation_config["eos_token_id"].nil? && generation_config["min_length"] > 0
        processors.push(MinLengthLogitsProcessor.new(generation_config["min_length"], generation_config["eos_token_id"]))
      end

      if !generation_config["min_new_tokens"].nil? && !generation_config["eos_token_id"].nil? && generation_config["min_new_tokens"] > 0
        processors.push(MinNewTokensLengthLogitsProcessor.new(
          input_ids_seq_length,
          generation_config["min_new_tokens"],
          generation_config["eos_token_id"]
        ))
      end

      if !generation_config["forced_bos_token_id"].nil?
        processors.push(ForcedBOSTokenLogitsProcessor.new(generation_config["forced_bos_token_id"]))
      end

      if !generation_config["forced_eos_token_id"].nil?
        processors.push(ForcedEOSTokenLogitsProcessor.new(
          generation_config["max_length"],
          generation_config["forced_eos_token_id"]
        ))
      end

      if !generation_config["begin_suppress_tokens"].nil?
        raise Todo
      end

      if !generation_config["forced_decoder_ids"].nil?
        processors.push(ForceTokensLogitsProcessor.new(generation_config["forced_decoder_ids"]))
      end

      if !logits_processor.nil?
        processors.concat(logits_processor)
      end

      processors
    end

    def get_generation_config(generation_config)
      # Create empty generation config (contains defaults)
      # We pass `@config` so that if `eos_token_id` or `bos_token_id` exist in the model's config, we will use them
      gen_config = Utils::GenerationConfig.new(@config.to_h)

      # Apply model's generation config, if it exists
      if @generation_config
        gen_config.merge!(@generation_config)
      end

      # Finally, use any generation config specified by the user
      # when calling `generate`
      if !generation_config.nil?
        gen_config.merge!(generation_config)
      end

      gen_config
    end

    def seq2seq_forward(model_inputs)
      encoder_outputs = model_inputs[:encoder_outputs]
      past_key_values = model_inputs[:past_key_values]

      if !encoder_outputs
        # Encoder outputs are not given, so we must compute them.
        encoder_outputs = encoder_forward(model_inputs)[0]
      end
      decoder_feeds = {
        input_ids: model_inputs[:decoder_input_ids],
        encoder_hidden_states: encoder_outputs
      }
      use_cache_branch = !!past_key_values

      if @decoder_merged_session.inputs.map { |v| v[:name] }.include?("use_cache_branch")
        decoder_feeds[:use_cache_branch] = [use_cache_branch]
      end

      if @decoder_merged_session.inputs.map { |v| v[:name] }.include?("encoder_attention_mask")
        decoder_feeds[:encoder_attention_mask] = model_inputs[:attention_mask]
      end

      prepare_position_ids(@decoder_merged_session, decoder_feeds, use_cache_branch)
      add_past_key_values(decoder_feeds, past_key_values)

      decoder_results = session_run(@decoder_merged_session, decoder_feeds)
      decoder_results = @decoder_merged_session.outputs.map { |v| v[:name] }.zip(decoder_results).to_h
      logits = decoder_results["logits"]
      past_key_values = get_past_key_values(decoder_results, past_key_values)

      # Get cross attention and/or decoder attentions if they are present
      attns = get_attentions(decoder_results)

      Seq2SeqLMOutput.new(logits, past_key_values, encoder_outputs, *attns)
    end

    def prepare_position_ids(session, feeds, use_cache_branch)
      if !session.inputs.map { |v| v[:name] }.include?("position_ids")
        return
      end

      raise Todo
    end

    def get_past_key_values(decoder_results, past_key_values)
      pkvs = {}

      decoder_results.each_key do |name|
        if name.start_with?("present")
          new_name = name.sub("present", "past_key_values")

          if past_key_values && name.include?("encoder")
            # Optimization introduced by optimum to reuse past key values. So, we just replace the constant
            # outputs with the previous past key values.
            # https://github.com/huggingface/optimum/blob/0bf2c05fb7e1182b52d21b703cfc95fd9e4ea3dc/optimum/onnxruntime/base.py#L677-L704
            pkvs[new_name] = past_key_values[new_name]
          else
            pkvs[new_name] = decoder_results[name]
          end
        end
      end
      pkvs
    end

    def get_attentions(decoder_results)
      attns = {}

      ["cross_attentions", "decoder_attentions"].each do |attn_name|
        result = []
        decoder_results.each_key do |name|
          if name.start_with?(attn_name)
            index = name.split(".").pop
            result[index] = decoder_results[name]
          end
        end
        attns[attn_name] = result
      end
      attns
    end

    def add_past_key_values(decoder_feeds, past_key_values)
      if past_key_values
        decoder_feeds.merge!(past_key_values)
      else
        # TODO support batches (i.e., batch_size > 1)
        batch_size = 1

        if @config[:is_encoder_decoder] && (!@add_encoder_pkv.nil? ? @add_encoder_pkv : true)
          raise Todo
        elsif @config[:model_type] == "falcon"
          raise Todo
        elsif @config[:multi_query]
          raise Todo
        elsif @config[:model_type] == "bloom"
          raise Todo
        else
          dims = [batch_size, @num_heads, 0, @dim_kv]
          @num_layers.times do |i|
            # decoder_feeds["past_key_values.#{i}.key"] = new Tensor('float32', [], dims)
            # decoder_feeds["past_key_values.#{i}.value"] = new Tensor('float32', [], dims)
          end
        end
      end
    end

    def seq2seq_start_beams(input_token_ids, generation_config, num_output_tokens, inputs_attention_mask = nil)
      beams = []
      beam_id = 0

      requires_attention_mask = !@requires_attention_mask.nil? ? @requires_attention_mask : true

      # decoder_input_ids == output_token_ids
      decoder_input_ids =
        generation_config["decoder_input_ids"] ||
        generation_config["decoder_start_token_id"] ||
        generation_config["bos_token_id"] ||
        generation_config["eos_token_id"]

      if !decoder_input_ids.is_a?(Array)
        decoder_input_ids = [decoder_input_ids]
      end

      input_token_ids.each do |tokens|
        # TODO: Improve
        # Currently, just add back batch dimension.
        # In future, allow for true parallel execution
        tokens = [tokens]

        # Create beam
        start = {
          inputs: tokens,
          encoder_outputs: nil,
          prev_model_outputs: nil,

          output_token_ids: decoder_input_ids,
          done: false,
          score: 0,
          id: beam_id # assign unique id to beams
        }
        beam_id += 1

        if requires_attention_mask
          start[:attention_mask] = prepare_attention_mask(tokens)
        end

        beams << start
      end

      beams
    end

    def prepare_attention_mask(tokens)
      # Prepare attention mask
      pad_token_id = @config["pad_token_id"]
      eos_token_id = @config["eos_token_id"]
      if eos_token_id.is_a?(Integer)
        eos_token_id = [eos_token_id]
      end

      is_pad_token_in_inputs = !tokens.index(pad_token_id).nil?
      is_pad_token_not_equal_to_eos_token_id = eos_token_id.nil? || !eos_token_id.include?(pad_token_id)

      if is_pad_token_in_inputs && is_pad_token_not_equal_to_eos_token_id
        raise Todo
      else
        Utils.ones_like(tokens)
      end
    end

    def seq2seq_run_beam(beam)
      # TODO use MAIN_INPUT_NAME
      input_name = :pixel_values

      decoder_input_ids = beam[:output_token_ids]
      if beam[:prev_model_outputs]
        # After the first step, `prev_model_outputs` won't be null.
        # So, we cut decoder_input_ids if past is used
        # TODO
        # decoder_input_ids = decoder_input_ids.slice(-1)
      end

      # 1. Prepare
      model_inputs = {
        input_name => beam[:inputs],
        decoder_input_ids: [decoder_input_ids],
        encoder_outputs: beam[:encoder_outputs],
        past_key_values: beam[:prev_model_outputs] && beam[:prev_model_outputs][:past_key_values]
      }
      if beam[:attention_mask]
        model_inputs[:attention_mask] = beam[:attention_mask]
      end

      # 2. Run
      output = @forward.(model_inputs)

      # 3. Update
      beam[:prev_model_outputs] = output
      beam[:encoder_outputs] = output[:encoder_outputs]

      output
    end

    def seq2seq_update_beam(beam, new_token_id)
      beam[:output_token_ids] += [new_token_id]
    end

    def group_beams(beams)
      # Group beams by their ids
      groups = {}
      beams.each do |obj|
        if !groups[obj[:id]]
          groups[obj[:id]] = [obj]
        else
          groups[obj[:id]] << obj
        end
      end
      groups.values
    end

    def encoder_forward(model_inputs, output_names: nil)
      encoder_feeds = {}
      @session.inputs.each do |input|
        key = input[:name].to_sym
        encoder_feeds[key] = model_inputs[key]
      end
      if @session.inputs.any? { |v| v[:name] == "token_type_ids" } && !encoder_feeds[:token_type_ids]
        raise Todo
      end
      session_run(@session, encoder_feeds, output_names:)
    end

    def decoder_forward(model_inputs)
      raise Todo
    end

    def decoder_start_beams(input_token_ids, generation_config, num_output_tokens, inputs_attention_mask)
      raise Todo
    end

    def decoder_run_beam(beam)
      raise Todo
    end

    def decoder_update_beam(beam, new_token_id)
      raise Todo
    end

    def session_run(session, inputs, output_names: nil)
      checked_inputs = validate_inputs(session, inputs)
      begin
        output = session.run(output_names || @output_names, checked_inputs)
        output = replace_tensors(output)
        output
      rescue => e
        raise e
      end
    end

    # TODO
    def replace_tensors(obj)
      obj
    end

    # TODO
    def validate_inputs(session, inputs)
      inputs
    end

    def get_start_beams(input_token_ids, generation_config, num_output_tokens, inputs_attention_mask)
      @get_start_beams.(input_token_ids, generation_config, num_output_tokens, inputs_attention_mask)
    end

    def run_beam(beam)
      @run_beam.(beam)
    end

    def update_beam(beam, new_token_id)
      @update_beam.(beam, new_token_id)
    end
  end

  class BertPreTrainedModel < PreTrainedModel
  end

  class BertModel < BertPreTrainedModel
  end

  class BertForMaskedLM < BertPreTrainedModel
    def call(model_inputs)
      MaskedLMOutput.new(*super(model_inputs))
    end
  end

  class BertForSequenceClassification < BertPreTrainedModel
    def call(model_inputs)
      SequenceClassifierOutput.new(*super(model_inputs))
    end
  end

  class BertForTokenClassification < BertPreTrainedModel
    def call(model_inputs)
      TokenClassifierOutput.new(*super(model_inputs))
    end
  end

  class NomicBertPreTrainedModel < PreTrainedModel
  end

  class NomicBertModel < NomicBertPreTrainedModel
  end

  class DebertaV2PreTrainedModel < PreTrainedModel
  end

  class DebertaV2Model < DebertaV2PreTrainedModel
  end

  class DistilBertPreTrainedModel < PreTrainedModel
  end

  class DistilBertModel < DistilBertPreTrainedModel
  end

  class DistilBertForSequenceClassification < DistilBertPreTrainedModel
    def call(model_inputs)
      SequenceClassifierOutput.new(*super(model_inputs))
    end
  end

  class DistilBertForQuestionAnswering < DistilBertPreTrainedModel
    def call(model_inputs)
      QuestionAnsweringModelOutput.new(*super(model_inputs))
    end
  end

  class MPNetPreTrainedModel < PreTrainedModel
  end

  class MPNetModel < MPNetPreTrainedModel
  end

  class T5PreTrainedModel < PreTrainedModel
  end

  class T5Model < T5PreTrainedModel
  end

  class T5ForConditionalGeneration < T5PreTrainedModel
    def initialize(config, session, decoder_merged_session, generation_config)
      super(config, session)
      @decoder_merged_session = decoder_merged_session
      @generation_config = generation_config

      @num_decoder_layers = @config[:num_decoder_layers]
      @num_decoder_heads = @config[:num_heads]
      @decoder_dim_kv = @config[:d_kv]

      @num_encoder_layers = @config[:num_layers]
      @num_encoder_heads = @config[:num_heads]
      @encoder_dim_kv = @config[:d_kv]
    end
  end

  class BartPretrainedModel < PreTrainedModel
  end

  class BartModel < BartPretrainedModel
  end

  class BartForSequenceClassification < BartPretrainedModel
    def call(model_inputs)
      SequenceClassifierOutput.new(*super(model_inputs))
    end
  end

  class RobertaPreTrainedModel < PreTrainedModel
  end

  class RobertaModel < RobertaPreTrainedModel
  end

  class RobertaForMaskedLM < RobertaPreTrainedModel
    def call(model_inputs)
      MaskedLMOutput.new(*super(model_inputs))
    end
  end

  class XLMRobertaPreTrainedModel < PreTrainedModel
  end

  class XLMRobertaModel < XLMRobertaPreTrainedModel
  end

  class XLMRobertaForSequenceClassification < XLMRobertaPreTrainedModel
    def call(model_inputs)
      SequenceClassifierOutput.new(*super(model_inputs))
    end
  end

  class ViTPreTrainedModel < PreTrainedModel
  end

  class ViTModel < ViTPreTrainedModel
  end

  class ViTForImageClassification < ViTPreTrainedModel
    def call(model_inputs)
      SequenceClassifierOutput.new(*super(model_inputs))
    end
  end

  class CLIPPreTrainedModel < PreTrainedModel
  end

  class CLIPModel < CLIPPreTrainedModel
  end

  class GPT2PreTrainedModel < PreTrainedModel
    attr_reader :num_heads, :num_layers, :dim_kv

    def initialize(config, session, generation_config)
      super(config, session)
      @generation_config = generation_config

      # config doesn't contain pad_token_id, so we assume it is the eos_token_id
      @config["pad_token_id"] = @config["eos_token_id"]

      @num_heads = @config["n_head"]
      @num_layers = @config["n_layer"]
      @dim_kv = @config["n_embd"] / @num_heads.to_f
    end
  end

  class GPT2Model < GPT2PreTrainedModel
  end

  class GPT2LMHeadModel < GPT2PreTrainedModel
  end

  class OwlViTPreTrainedModel < PreTrainedModel
  end

  class OwlViTModel < OwlViTPreTrainedModel
  end

  class OwlViTForObjectDetection < OwlViTPreTrainedModel
  end

  class DetrPreTrainedModel < PreTrainedModel
  end

  class DetrModel < DetrPreTrainedModel
  end

  class DetrForObjectDetection < DetrPreTrainedModel
    def call(model_inputs)
      DetrObjectDetectionOutput.new(*super(model_inputs))
    end
  end

  class DetrForSegmentation < DetrPreTrainedModel
    def call(model_inputs)
      DetrSegmentationOutput.new(*super(model_inputs))
    end
  end

  class DPTPreTrainedModel < PreTrainedModel
  end

  class DPTModel < DPTPreTrainedModel
  end

  class DPTForDepthEstimation < DPTPreTrainedModel
  end

  class VisionEncoderDecoderModel < PreTrainedModel
    MAIN_INPUT_NAME = :pixel_values

    def initialize(config, session, decoder_merged_session, generation_config)
      super(config, session)
      @decoder_merged_session = decoder_merged_session
      @generation_config = generation_config

      # Extract configs
      encoder_config = @config["encoder"]
      decoder_config = @config["decoder"]

      # Validate encoder
      encoder_model_type = encoder_config["model_type"]
      encoder_model = MODEL_MAPPING_NAMES_ENCODER_ONLY[encoder_model_type] || MODEL_MAPPING_NAMES_ENCODER_DECODER[encoder_model_type]
      if !encoder_model
        warn "Model type for encoder '#{encoder_model_type}' not found, assuming encoder-only architecture. Please report this."
      end

      # Validate decoder
      decoder_model = MODEL_WITH_LM_HEAD_MAPPING_NAMES[decoder_config["model_type"]]
      if !decoder_model
        raise Error, "Unable to construct `VisionEncoderDecoder` due to unsupported decoder: \"#{decoder_config["model_type"]}\""
      end

      decoder_model_class = decoder_model[1]
      decoder = decoder_model_class.new(decoder_config, decoder_merged_session, generation_config)

      @add_encoder_pkv = decoder.config.include?("num_decoder_layers")
      if @add_encoder_pkv
        raise Todo
      else
        # Decoder is a decoder-only model
        @num_layers = decoder.num_layers
        @num_heads = decoder.num_heads
        @dim_kv = decoder.dim_kv
      end
    end
  end

  MODEL_MAPPING_NAMES_ENCODER_ONLY = {
    "bert" => ["BertModel", BertModel],
    "nomic_bert" => ["NomicBertModel", NomicBertModel],
    "deberta-v2" => ["DebertaV2Model", DebertaV2Model],
    "mpnet" => ["MPNetModel", MPNetModel],
    "distilbert" => ["DistilBertModel", DistilBertModel],
    "roberta" => ["RobertaModel", RobertaModel],
    "xlm-roberta" => ["XLMRobertaModel", XLMRobertaModel],
    "clip" => ["CLIPModel", CLIPModel],
    "detr" => ["DetrModel", DetrModel],
    "vit" => ["ViTModel", ViTModel],
    "owlvit" => ["OwlViTModel", OwlViTModel]
  }

  MODEL_MAPPING_NAMES_ENCODER_DECODER = {
    "bart" => ["BartModel", BartModel]
  }

  MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = {
    "bert" => ["BertForSequenceClassification", BertForSequenceClassification],
    "distilbert" => ["DistilBertForSequenceClassification", DistilBertForSequenceClassification],
    "xlm-roberta" => ["XLMRobertaForSequenceClassification", XLMRobertaForSequenceClassification],
    "bart" => ["BartForSequenceClassification", BartForSequenceClassification]
  }

  MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = {
    "bert" => ["BertForTokenClassification", BertForTokenClassification]
  }

  MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = {
    "t5" => ["T5ForConditionalGeneration", T5ForConditionalGeneration]
  }

  MODEL_WITH_LM_HEAD_MAPPING_NAMES = {
    "gpt2" => ["GPT2LMHeadModel", GPT2LMHeadModel]
  }

  MODEL_FOR_MASKED_LM_MAPPING_NAMES = {
    "bert" => ["BertForMaskedLM", BertForMaskedLM],
    "roberta" => ["RobertaForMaskedLM", RobertaForMaskedLM]
  }

  MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = {
    "distilbert" => ["DistilBertForQuestionAnswering", DistilBertForQuestionAnswering]
  }

  MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {
    "vision-encoder-decoder" => ["VisionEncoderDecoderModel", VisionEncoderDecoderModel]
  }

  MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = {
    "vit" => ["ViTForImageClassification", ViTForImageClassification]
  }

  MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = {
    "detr" => ["DetrForObjectDetection", DetrForObjectDetection]
  }

  MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES = {
    "owlvit" => ["OwlViTForObjectDetection", OwlViTForObjectDetection]
  }

  MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = {
    "detr" => ["DetrForSegmentation", DetrForSegmentation]
  }

  MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = {
  }

  MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = {
    "dpt" => ["DPTForDepthEstimation", DPTForDepthEstimation]
  }

  MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES = {
  }

  MODEL_CLASS_TYPE_MAPPING = [
    [MODEL_MAPPING_NAMES_ENCODER_ONLY, MODEL_TYPES[:EncoderOnly]],
    [MODEL_MAPPING_NAMES_ENCODER_DECODER, MODEL_TYPES[:EncoderDecoder]],
    [MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]],
    [MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]],
    [MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES, MODEL_TYPES[:Seq2Seq]],
    [MODEL_WITH_LM_HEAD_MAPPING_NAMES, MODEL_TYPES[:DecoderOnly]],
    [MODEL_FOR_MASKED_LM_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]],
    [MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]],
    [MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES, MODEL_TYPES[:Vision2Seq]],
    [MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]],
    [MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]],
    [MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]],
    [MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]],
    [MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]],
    [MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]],
    [MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES, MODEL_TYPES[:EncoderOnly]]
  ]

  MODEL_CLASS_TYPE_MAPPING.each do |mappings, type|
    mappings.values.each do |name, model|
      MODEL_TYPE_MAPPING[name] = type
      MODEL_CLASS_TO_NAME_MAPPING[model] = name
      MODEL_NAME_TO_CLASS_MAPPING[name] = model
    end
  end

  class AutoModel < PretrainedMixin
    MODEL_CLASS_MAPPINGS = MODEL_CLASS_TYPE_MAPPING.map { |x| x[0] }
    BASE_IF_FAIL = true
  end

  class AutoModelForSequenceClassification < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES]
  end

  class AutoModelForTokenClassification < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES]
  end

  class AutoModelForSeq2SeqLM < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES]
  end

  class AutoModelForMaskedLM < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_MASKED_LM_MAPPING_NAMES]
  end

  class AutoModelForQuestionAnswering < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES]
  end

  class AutoModelForVision2Seq < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES]
  end

  class AutoModelForImageClassification < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES]
  end

  class AutoModelForImageSegmentation < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES]
  end

  class AutoModelForSemanticSegmentation < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES]
  end

  class AutoModelForObjectDetection < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES]
  end

  class AutoModelForZeroShotObjectDetection < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES]
  end

  class AutoModelForDepthEstimation < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES]
  end

  class AutoModelForImageFeatureExtraction < PretrainedMixin
    MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES]
  end

  class ModelOutput
    def [](key)
      instance_variable_get("@#{key}")
    end
  end

  class Seq2SeqLMOutput < ModelOutput
    def initialize(logits, past_key_values, encoder_outputs, decoder_attentions = nil, cross_attentions = nil)
      super()
      @logits = logits
      @past_key_values = past_key_values
      @encoder_outputs = encoder_outputs
      @decoder_attentions = decoder_attentions
      @cross_attentions = cross_attentions
    end
  end

  class SequenceClassifierOutput < ModelOutput
    attr_reader :logits

    def initialize(logits)
      super()
      @logits = logits
    end
  end

  class TokenClassifierOutput < ModelOutput
    attr_reader :logits

    def initialize(logits)
      super()
      @logits = logits
    end
  end

  class MaskedLMOutput < ModelOutput
    attr_reader :logits

    def initialize(logits)
      super()
      @logits = logits
    end
  end

  class QuestionAnsweringModelOutput < ModelOutput
    attr_reader :start_logits, :end_logits

    def initialize(start_logits, end_logits)
      super()
      @start_logits = start_logits
      @end_logits = end_logits
    end
  end

  class DetrObjectDetectionOutput < ModelOutput
    attr_reader :logits, :pred_boxes

    def initialize(logits, pred_boxes)
      super()
      @logits = logits
      @pred_boxes = pred_boxes
    end
  end

  class DetrSegmentationOutput < ModelOutput
    attr_reader :logits, :pred_boxes, :pred_masks

    def initialize(logits, pred_boxes, pred_masks)
      super()
      @logits = logits
      @pred_boxes = pred_boxes
      @pred_masks = pred_masks
    end
  end
end
