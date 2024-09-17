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
        raise Todo
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

    private

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
