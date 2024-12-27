module Informers
  module Utils
    module Hub
      class FileResponse
        attr_reader :exists, :status

        def initialize(file_path)
          @file_path = file_path

          @exists = File.exist?(file_path)
          if @exists
            @status = ["200", "OK"]
          else
            @status = ["404", "Not Found"]
          end
        end

        def read
          File.binread(@file_path)
        end
      end

      def self.is_valid_url(string, protocols = nil, valid_hosts = nil)
        begin
          url = URI.parse(string)
        rescue
          return false
        end
        if protocols && !protocols.include?(url.scheme)
          return false
        end
        if valid_hosts && !valid_hosts.include?(url.host)
          return false
        end
        true
      end

      def self.get_file(url_or_path, progress_callback = nil, progress_info = {})
        if !is_valid_url(url_or_path, ["http", "https"])
          raise Error, "Invalid url"
        else
          headers = {}
          headers["User-Agent"] = "informers/#{VERSION};"

          # Check whether we are making a request to the Hugging Face Hub.
          is_hfurl = is_valid_url(url_or_path, ["http", "https"], ["huggingface.co", "hf.co"])
          if is_hfurl
            # If an access token is present in the environment variables,
            # we add it to the request headers.
            token = ENV["HF_TOKEN"]
            if token
              headers["Authorization"] = "Bearer #{token}"
            end
          end
          options = {}
          if progress_callback
            total_size = nil
            options[:content_length_proc] = lambda do |size|
              total_size = size
              Utils.dispatch_callback(progress_callback, {status: "download"}.merge(progress_info).merge(total_size: size))
            end
            options[:progress_proc] = lambda do |size|
              Utils.dispatch_callback(progress_callback, {status: "progress"}.merge(progress_info).merge(size: size, total_size: total_size))
            end
          end
          URI.parse(url_or_path).open(**headers, **options)
        end
      end

      class FileCache
        attr_reader :path

        def initialize(path)
          @path = path
        end

        def match(request)
          file_path = resolve_path(request)
          file = FileResponse.new(file_path)

          file if file.exists
        end

        def put(request, response)
          output_path = resolve_path(request)

          begin
            tmp_path = "#{output_path}.incomplete"
            FileUtils.mkdir_p(File.dirname(output_path))
            File.open(tmp_path, "wb") do |f|
              while !response.eof?
                f.write(response.read(1024 * 1024))
              end
            end
            FileUtils.move(tmp_path, output_path)
          rescue => e
            warn "An error occurred while writing the file to cache: #{e}"
          end
        end

        def resolve_path(request)
          File.join(@path, request)
        end
      end

      def self.try_cache(cache, *names)
        names.each do |name|
          begin
            result = cache.match(name)
            return result if result
          rescue
            next
          end
        end
        nil
      end

      def self.get_model_file(path_or_repo_id, filename, fatal = true, **options)
        # Initiate file retrieval
        Utils.dispatch_callback(options[:progress_callback], {
          status: "initiate",
          name: path_or_repo_id,
          file: filename
        })

        # If `cache_dir` is not specified, use the default cache directory
        cache = FileCache.new(options[:cache_dir] || Informers.cache_dir)

        revision = options[:revision] || "main"

        request_url = path_join(path_or_repo_id, filename)

        remote_url = path_join(
          Informers.remote_host,
          Informers.remote_path_template
            .gsub("{model}", path_or_repo_id)
            .gsub("{revision}", URI.encode_www_form_component(revision)),
          filename
        )

        # Choose cache key for filesystem cache
        # When using the main revision (default), we use the request URL as the cache key.
        # If a specific revision is requested, we account for this in the cache key.
        fs_cache_key = revision == "main" ? request_url : path_join(path_or_repo_id, revision, filename)

        proposed_cache_key = fs_cache_key

        resolved_path = cache.resolve_path(proposed_cache_key)

        # Whether to cache the final response in the end.
        to_cache_response = false

        # A caching system is available, so we try to get the file from it.
        response = try_cache(cache, proposed_cache_key)

        cache_hit = !response.nil?

        if response.nil?
          # File is not cached, so we perform the request

          if response.nil? || response.status[0] == "404"
            # File not found locally. This means either:
            # - The user has disabled local file access (`Informers.allow_local_models = false`)
            # - the path is a valid HTTP url (`response.nil?`)
            # - the path is not a valid HTTP url and the file is not present on the file system or local server (`response.status[0] == "404"`)

            if options[:local_files_only] || !Informers.allow_remote_models
              # User requested local files only, but the file is not found locally.
              if fatal
                raise Error, "`local_files_only: true` or `Informers.allow_remote_models = false` and file was not found locally at #{resolved_path.inspect}."
              else
                # File not found, but this file is optional.
                # TODO in future, cache the response?
                return nil
              end
            end

            progress_info = {
              name: path_or_repo_id,
              file: filename
            }

            # File not found locally, so we try to download it from the remote server
            response = get_file(remote_url, options[:progress_callback], progress_info)

            if response.status[0] != "200"
              # should not happen
              raise Todo
            end

            # Success! We use the proposed cache key from earlier
            cache_key = proposed_cache_key
          end

          to_cache_response = cache && !response.is_a?(FileResponse) && response.status[0] == "200"
        end

        if to_cache_response && cache_key && cache.match(cache_key).nil?
          cache.put(cache_key, response)
        end

        Utils.dispatch_callback(options[:progress_callback], {
          status: "done",
          name: path_or_repo_id,
          file: filename,
          cache_hit: cache_hit
        })

        resolved_path
      end

      def self.get_model_json(model_path, file_name, fatal = true, **options)
        buffer = get_model_file(model_path, file_name, fatal, **options)
        if buffer.nil?
          # Return empty object
          return {}
        end

        JSON.load_file(buffer)
      end

      def self.path_join(*parts)
        parts = parts.map.with_index do |part, index|
          if index != 0
            part = part.delete_prefix("/")
          end
          if index != parts.length - 1
            part = part.delete_suffix("/")
          end
          part
        end
        parts.join("/")
      end

      def self.display_progress(filename, width, size, expected_size)
        bar_width = [width - (filename.length + 3), 1].max
        progress = expected_size ? size / expected_size.to_f : 0
        done = (progress * bar_width).round
        not_done = bar_width - done
        "#{filename} |#{"█" * done}#{" " * not_done}|"
      end
    end
  end
end
