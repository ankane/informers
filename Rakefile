require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = FileList["test/**/*_test.rb"].exclude("test/model_test.rb")
end
