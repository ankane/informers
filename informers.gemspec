require_relative "lib/informers/version"

Gem::Specification.new do |spec|
  spec.name          = "informers"
  spec.version       = Informers::VERSION
  spec.summary       = "State-of-the-art natural language processing for Ruby"
  spec.homepage      = "https://github.com/ankane/informers"
  spec.license       = "Apache-2.0"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@chartkick.com"

  spec.files         = Dir["*.{md,txt}", "{lib,vendor}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.5"

  spec.add_dependency "blingfire", ">= 0.1.3"
  spec.add_dependency "onnxruntime"
end
