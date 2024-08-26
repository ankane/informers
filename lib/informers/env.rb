module Informers
  CACHE_HOME = ENV.fetch("XDG_CACHE_HOME", File.join(ENV.fetch("HOME"), ".cache"))
  DEFAULT_CACHE_DIR = File.expand_path(File.join(CACHE_HOME, "informers"))

  class << self
    attr_accessor :allow_remote_models, :remote_host, :remote_path_template, :cache_dir
  end

  self.allow_remote_models = ENV["INFORMERS_OFFLINE"].to_s.empty?
  self.remote_host = "https://huggingface.co/"
  self.remote_path_template = "{model}/resolve/{revision}/"

  self.cache_dir = DEFAULT_CACHE_DIR
end
