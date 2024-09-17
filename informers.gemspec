require_relative "lib/informers/version"

Gem::Specification.new do |spec|
  spec.name          = "informers"
  spec.version       = Informers::VERSION
  spec.summary       = "Fast transformer inference for Ruby"
  spec.homepage      = "https://github.com/ankane/informers"
  spec.license       = "Apache-2.0"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 3.1"

  spec.add_dependency "onnxruntime", ">= 0.9"
  spec.add_dependency "tokenizers", ">= 0.5.3"
end
