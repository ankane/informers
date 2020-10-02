require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"

class Minitest::Test
  def models_path
    ENV.fetch("MODELS_PATH")
  end
end
