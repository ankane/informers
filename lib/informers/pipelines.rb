module Informers
  class Pipeline
    def initialize(task:, model:, tokenizer: nil, processor: nil)
      super()
      @task = task
      @model = model
      @tokenizer = tokenizer
      @processor = processor
    end
  end

  class TextClassificationPipeline < Pipeline
    def call(texts, top_k: 1)
      # Run tokenization
      model_inputs = @tokenizer.(texts,
        padding: true,
        truncation: true
      )

      # Run model
      outputs = @model.(model_inputs)

      function_to_apply =
        if @model.config.problem_type == "multi_label_classification"
          ->(batch) { Utils.sigmoid(batch) }
        else
          ->(batch) { Utils.softmax(batch) } # single_label_classification (default)
        end

      id2label = @model.config.id2label

      to_return = []
      outputs.logits.each do |batch|
        output = function_to_apply.(batch)
        scores = Utils.get_top_items(output, top_k)

        vals = scores.map do |x|
          {
            label: id2label[x[0].to_s],
            score: x[1]
          }
        end
        if top_k == 1
          to_return.concat(vals)
        else
          to_return << vals
        end
      end

      texts.is_a?(Array) ? to_return : to_return[0]
    end
  end

  class TokenClassificationPipeline < Pipeline
    def call(
      texts,
      ignore_labels: ["O"],
      aggregation_strategy: "simple"
    )
      is_batched = texts.is_a?(Array)

      # Run tokenization
      model_inputs = @tokenizer.(is_batched ? texts : [texts],
        padding: true,
        truncation: true,
        return_offsets: true
      )

      # Run model
      outputs = @model.(model_inputs)

      logits = outputs.logits
      id2label = @model.config.id2label

      to_return = []
      logits.length.times do |i|
        ids = model_inputs[:input_ids][i]
        batch = logits[i]
        offsets = model_inputs[:offsets][i]

        # List of tokens that aren't ignored
        tokens = []
        batch.length.times do |j|
          token_data = batch[j]
          top_score_index = Utils.max(token_data)[1]

          entity = id2label ? id2label[top_score_index.to_s] : "LABEL_#{top_score_index}"
          if ignore_labels.include?(entity)
            # We predicted a token that should be ignored. So, we skip it.
            next
          end

          # TODO add option to keep special tokens?
          word = @tokenizer.decode([ids[j]], skip_special_tokens: true)
          if word == ""
            # Was a special token. So, we skip it.
            next
          end

          scores = Utils.softmax(token_data)

          tokens << {
            entity: entity,
            score: scores[top_score_index],
            index: j,
            word: word,
            start: offsets[j][0],
            end: offsets[j][1]
          }
        end

        case aggregation_strategy
        when "simple"
          tokens = group_entities(tokens)
        when "none"
          # do nothing
        else
          raise ArgumentError, "Invalid aggregation_strategy"
        end

        to_return << tokens
      end
      is_batched ? to_return : to_return[0]
    end

    def group_sub_entities(entities)
      # Get the first entity in the entity group
      entity = entities[0][:entity].split("-", 2)[-1]
      scores = entities.map { |entity| entity[:score] }
      tokens = entities.map { |entity| entity[:word] }

      entity_group = {
        entity_group: entity,
        score: scores.sum / scores.count.to_f,
        word: @tokenizer.convert_tokens_to_string(tokens),
        start: entities[0][:start],
        end: entities[-1][:end]
      }
      entity_group
    end

    def get_tag(entity_name)
      if entity_name.start_with?("B-")
        bi = "B"
        tag = entity_name[2..]
      elsif entity_name.start_with?("I-")
        bi = "I"
        tag = entity_name[2..]
      else
        # It's not in B-, I- format
        # Default to I- for continuation.
        bi = "I"
        tag = entity_name
      end
      [bi, tag]
    end

    def group_entities(entities)
      entity_groups = []
      entity_group_disagg = []

      entities.each do |entity|
        if entity_group_disagg.empty?
          entity_group_disagg << entity
          next
        end

        # If the current entity is similar and adjacent to the previous entity,
        # append it to the disaggregated entity group
        # The split is meant to account for the "B" and "I" prefixes
        # Shouldn't merge if both entities are B-type
        bi, tag = get_tag(entity[:entity])
        _last_bi, last_tag = get_tag(entity_group_disagg[-1][:entity])

        if tag == last_tag && bi != "B"
          # Modify subword type to be previous_type
          entity_group_disagg << entity
        else
          # If the current entity is different from the previous entity
          # aggregate the disaggregated entity group
          entity_groups << group_sub_entities(entity_group_disagg)
          entity_group_disagg = [entity]
        end
      end
      if entity_group_disagg.any?
        # it's the last entity, add it to the entity groups
        entity_groups << group_sub_entities(entity_group_disagg)
      end

      entity_groups
    end
  end

  class QuestionAnsweringPipeline < Pipeline
    def call(question, context, top_k: 1)
      # Run tokenization
      inputs = @tokenizer.(question,
        text_pair: context,
        padding: true,
        truncation: true,
        return_offsets: true
      )

      output = @model.(inputs)

      to_return = []
      output.start_logits.length.times do |j|
        ids = inputs[:input_ids][j]
        sep_index = ids.index(@tokenizer.sep_token_id)
        offsets = inputs[:offsets][j]

        s1 = Utils.softmax(output.start_logits[j])
          .map.with_index
          .select { |x| x[1] > sep_index }
        e1 = Utils.softmax(output.end_logits[j])
          .map.with_index
          .select { |x| x[1] > sep_index }

        options = s1.product(e1)
          .select { |x| x[0][1] <= x[1][1] }
          .map { |x| [x[0][1], x[1][1], x[0][0] * x[1][0]] }
          .sort_by { |v| -v[2] }

        [options.length, top_k].min.times do |k|
          start, end_, score = options[k]

          answer_tokens = ids.slice(start, end_ + 1)

          answer = @tokenizer.decode(answer_tokens,
            skip_special_tokens: true
          )

          to_return << {
            answer:,
            score:,
            start: offsets[start][0],
            end: offsets[end_][1]
          }
        end
      end

      question.is_a?(Array) ? to_return : to_return[0]
    end
  end

  class FillMaskPipeline < Pipeline
    def call(texts, top_k: 5)
      model_inputs = @tokenizer.(texts, padding: true, truncation: true)
      outputs = @model.(model_inputs)

      to_return = []
      model_inputs[:input_ids].each_with_index do |ids, i|
        mask_token_index = ids.index(@tokenizer.mask_token_id)

        if mask_token_index.nil?
          raise ArgumentError, "Mask token (#{@tokenizer.mask_token}) not found in text."
        end
        logits = outputs.logits[i]
        item_logits = logits[mask_token_index]

        scores = Utils.get_top_items(Utils.softmax(item_logits), top_k)

        to_return <<
          scores.map do |x|
            sequence = ids.dup
            sequence[mask_token_index] = x[0]

            {
              score: x[1],
              token: x[0],
              token_str: @tokenizer.id_to_token(x[0]),
              sequence: @tokenizer.decode(sequence, skip_special_tokens: true)
            }
          end
      end
      texts.is_a?(Array) ? to_return : to_return[0]
    end
  end

  class ZeroShotClassificationPipeline < Pipeline
    def initialize(**options)
      super(**options)

      @label2id = @model.config.label2id.transform_keys(&:downcase)

      @entailment_id = @label2id["entailment"]
      if @entailment_id.nil?
        warn "Could not find 'entailment' in label2id mapping. Using 2 as entailment_id."
        @entailment_id = 2
      end

      @contradiction_id = @label2id["contradiction"] || @label2id["not_entailment"]
      if @contradiction_id.nil?
        warn "Could not find 'contradiction' in label2id mapping. Using 0 as contradiction_id."
        @contradiction_id = 0
      end
    end

    def call(texts, candidate_labels, hypothesis_template: "This example is {}.", multi_label: false)
      is_batched = texts.is_a?(Array)
      if !is_batched
        texts = [texts]
      end
      if !candidate_labels.is_a?(Array)
        candidate_labels = [candidate_labels]
      end

      # Insert labels into hypothesis template
      hypotheses = candidate_labels.map { |x| hypothesis_template.sub("{}", x) }

      # How to perform the softmax over the logits:
      #  - true:  softmax over the entailment vs. contradiction dim for each label independently
      #  - false: softmax the "entailment" logits over all candidate labels
      softmax_each = multi_label || candidate_labels.length == 1

      to_return = []
      texts.each do |premise|
        entails_logits = []

        hypotheses.each do |hypothesis|
          inputs = @tokenizer.(
            premise,
            text_pair: hypothesis,
            padding: true,
            truncation: true
          )
          outputs = @model.(inputs)

          if softmax_each
            entails_logits << [
              outputs.logits[0][@contradiction_id],
              outputs.logits[0][@entailment_id]
            ]
          else
            entails_logits << outputs.logits[0][@entailment_id]
          end
        end

        scores =
          if softmax_each
            entails_logits.map { |x| Utils.softmax(x)[1] }
          else
            Utils.softmax(entails_logits)
          end

        # Sort by scores (desc) and return scores with indices
        scores_sorted = scores.map.with_index { |x, i| [x, i] }.sort_by { |v| -v[0] }

        to_return << {
          sequence: premise,
          labels: scores_sorted.map { |x| candidate_labels[x[1]] },
          scores: scores_sorted.map { |x| x[0] }
        }
      end
      is_batched ? to_return : to_return[0]
    end
  end

  class FeatureExtractionPipeline < Pipeline
    def call(
      texts,
      pooling: "none",
      normalize: false,
      quantize: false,
      precision: "binary",
      model_output: nil
    )
      # Run tokenization
      model_inputs = @tokenizer.(texts,
        padding: true,
        truncation: true
      )
      model_options = {}

      if !model_output.nil?
        model_options[:output_names] = Array(model_output)
      elsif @model.instance_variable_get(:@output_names) == ["token_embeddings"] && pooling == "mean" && normalize
        # optimization for sentence-transformers/all-MiniLM-L6-v2
        model_options[:output_names] = ["sentence_embedding"]
        pooling = "none"
        normalize = false
      end

      # Run model
      outputs = @model.(model_inputs, **model_options)

      # TODO improve
      result =
        if outputs.is_a?(Array)
          # TODO show returned instead of all
          output_names = @model.instance_variable_get(:@session).outputs.map { |v| v[:name] }
          raise Error, "unexpected outputs: #{output_names}" if outputs.size != 1
          outputs[0]
        else
          outputs.logits
        end

      case pooling
      when "none"
        # Skip pooling
      when "mean"
        result = Utils.mean_pooling(result, model_inputs[:attention_mask])
      when "cls"
        result = result.map(&:first)
      else
        # TODO raise ArgumentError in 2.0
        raise Error, "Pooling method '#{pooling}' not supported."
      end

      if normalize
        result = Utils.normalize(result)
      end

      if quantize
        result = quantize_embeddings(result, precision)
      end

      texts.is_a?(Array) ? result : result[0]
    end
  end

  class EmbeddingPipeline < FeatureExtractionPipeline
    def call(
      texts,
      pooling: "mean",
      normalize: true,
      model_output: nil
    )
      super(texts, pooling:, normalize:, model_output:)
    end
  end

  class RerankingPipeline < Pipeline
    def call(
      query,
      documents,
      return_documents: false,
      top_k: nil
    )
      model_inputs = @tokenizer.([query] * documents.size,
        text_pair: documents,
        padding: true,
        truncation: true
      )

      outputs = @model.(model_inputs)

      result =
        Utils.sigmoid(outputs[0].map(&:first))
          .map.with_index { |s, i| {doc_id: i, score: s} }
          .sort_by { |v| -v[:score] }

      if return_documents
        result.each do |v|
          v[:text] = documents[v[:doc_id]]
        end
      end

      top_k ? result.first(top_k) : result
    end
  end

  SUPPORTED_TASKS = {
    "text-classification" => {
      tokenizer: AutoTokenizer,
      pipeline: TextClassificationPipeline,
      model: AutoModelForSequenceClassification,
      default: {
        model: "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
      },
      type: "text"
    },
    "token-classification" => {
      tokenizer: AutoTokenizer,
      pipeline: TokenClassificationPipeline,
      model: AutoModelForTokenClassification,
      default: {
        model: "Xenova/bert-base-multilingual-cased-ner-hrl"
      },
      type: "text"
    },
    "question-answering" => {
      tokenizer: AutoTokenizer,
      pipeline: QuestionAnsweringPipeline,
      model: AutoModelForQuestionAnswering,
      default: {
        model: "Xenova/distilbert-base-cased-distilled-squad"
      },
      type: "text"
    },
    "fill-mask" => {
      tokenizer: AutoTokenizer,
      pipeline: FillMaskPipeline,
      model: AutoModelForMaskedLM,
      default: {
        model: "Xenova/bert-base-uncased"
      },
      type: "text"
    },
    "zero-shot-classification" => {
      tokenizer: AutoTokenizer,
      pipeline: ZeroShotClassificationPipeline,
      model: AutoModelForSequenceClassification,
      default: {
        model: "Xenova/distilbert-base-uncased-mnli"
      },
      type: "text"
    },
    "feature-extraction" => {
      tokenizer: AutoTokenizer,
      pipeline: FeatureExtractionPipeline,
      model: AutoModel,
      default: {
        model: "Xenova/all-MiniLM-L6-v2"
      },
      type: "text"
    },
    "embedding" => {
      tokenizer: AutoTokenizer,
      pipeline: EmbeddingPipeline,
      model: AutoModel,
      default: {
        model: "sentence-transformers/all-MiniLM-L6-v2"
      },
      type: "text"
    },
    "reranking" => {
      tokenizer: AutoTokenizer,
      pipeline: RerankingPipeline,
      model: AutoModel,
      default: {
        model: "mixedbread-ai/mxbai-rerank-base-v1"
      },
      type: "text"
    }
  }

  TASK_ALIASES = {
    "sentiment-analysis" => "text-classification",
    "ner" => "token-classification"
  }

  DEFAULT_PROGRESS_CALLBACK = lambda do |msg|
    stream = $stderr
    tty = stream.tty?
    width = tty ? stream.winsize[1] : 80

    if msg[:status] == "progress" && tty
      stream.print "\r#{Utils::Hub.display_progress(msg[:file], width, msg[:size], msg[:total_size])}"
    elsif msg[:status] == "done" && !msg[:cache_hit]
      if tty
        stream.puts
      else
        stream.puts Utils::Hub.display_progress(msg[:file], width, 1, 1)
      end
    end
  end

  NO_DEFAULT = Object.new

  class << self
    def pipeline(
      task,
      model = nil,
      quantized: NO_DEFAULT,
      progress_callback: DEFAULT_PROGRESS_CALLBACK,
      config: nil,
      cache_dir: nil,
      local_files_only: false,
      revision: "main",
      model_file_name: nil
    )
      # Apply aliases
      task = TASK_ALIASES[task] || task

      if quantized == NO_DEFAULT
        # TODO move default to task class
        quantized = ["text-classification", "token-classification", "question-answering", "feature-extraction"].include?(task)
      end

      # Get pipeline info
      pipeline_info = SUPPORTED_TASKS[task.split("_", 1)[0]]
      if !pipeline_info
        raise Error, "Unsupported pipeline: #{task}. Must be one of #{SUPPORTED_TASKS.keys}"
      end

      # Use model if specified, otherwise, use default
      if !model
        model = pipeline_info[:default][:model]
        warn "No model specified. Using default model: #{model.inspect}."
      end

      pretrained_options = {
        quantized:,
        progress_callback:,
        config:,
        cache_dir:,
        local_files_only:,
        revision:,
        model_file_name:
      }

      classes = {
        tokenizer: pipeline_info[:tokenizer],
        model: pipeline_info[:model],
        processor: pipeline_info[:processor]
      }

      # Load model, tokenizer, and processor (if they exist)
      results = load_items(classes, model, pretrained_options)
      results[:task] = task

      if model == "sentence-transformers/all-MiniLM-L6-v2"
        results[:model].instance_variable_set(:@output_names, ["token_embeddings"])
      end

      Utils.dispatch_callback(progress_callback, {
        status: "ready",
        task: task,
        model: model
      })

      pipeline_class = pipeline_info.fetch(:pipeline)
      pipeline_class.new(**results)
    end

    private

    def load_items(mapping, model, pretrained_options)
      result = {}

      mapping.each do |name, cls|
        next if !cls

        if cls.is_a?(Array)
          raise Todo
        else
          result[name] = cls.from_pretrained(model, **pretrained_options)
        end
      end

      result
    end
  end
end
