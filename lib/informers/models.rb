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
        raise Todo
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
      _max_output_tokens = num_output_tokens + (generation_config[:max_new_tokens] || Float::INFINITY)

      # Only use max length if max_new_tokens is not provided
      _use_max_length = generation_config[:max_length].is_a?(Integer) && generation_config[:max_new_tokens].nil?
      _sampler = Utils::Sampler.get_sampler(generation_config)

      _beams = get_start_beams(inputs, generation_config, num_output_tokens, inputs_attention_mask)

      raise Todo
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
      raise Todo
    end

    def seq2seq_start_beams(input_token_ids, generation_config, num_output_tokens, inputs_attention_mask = nil)
      raise Todo
    end

    def seq2seq_run_beam(beam)
      raise Todo
    end

    def seq2seq_update_beam(beam, new_token_id)
      raise Todo
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

    def session_run(session, inputs, output_names:)
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
    MAIN_INPUT_NAME = "pixel_values"

    def initialize(config, session, decoder_merged_session, generation_config)
      super(config, session)
      @decoder_merged_session = decoder_merged_session
      @generation_config = generation_config

      raise Todo
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
