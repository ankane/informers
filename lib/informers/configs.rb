module Informers
  class PretrainedConfig
    def initialize(config_json)
      @config_json = config_json
    end

    def [](key)
      @config_json[key.to_s]
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
