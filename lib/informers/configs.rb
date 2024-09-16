module Informers
  class PretrainedConfig
    attr_reader :model_type, :problem_type, :id2label, :label2id

    def initialize(config_json)
      @is_encoder_decoder = false

      @model_type = config_json["model_type"]
      @problem_type = config_json["problem_type"]
      @id2label = config_json["id2label"]
      @label2id = config_json["label2id"]
    end

    def [](key)
      instance_variable_get("@#{key}")
    end

    def self.from_pretrained(
      pretrained_model_name_or_path,
      progress_callback: nil,
      config: nil,
      cache_dir: nil,
      local_files_only: false,
      revision: "main",
      **kwargs
    )
      data = config || load_config(
        pretrained_model_name_or_path,
        progress_callback:,
        config:,
        cache_dir:,
        local_files_only:,
        revision:
      )
      new(data)
    end

    def self.load_config(pretrained_model_name_or_path, **options)
      info = Utils::Hub.get_model_json(pretrained_model_name_or_path, "config.json", true, **options)
      info
    end
  end

  class AutoConfig
    def self.from_pretrained(...)
      PretrainedConfig.from_pretrained(...)
    end
  end
end
