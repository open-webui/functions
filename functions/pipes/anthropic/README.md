# Anthropic Manifold Pipe

A simple Python interface to interact with Anthropic language models via the Open Web UI manifold pipe system.

---

## Features

- Supports all available Anthropic models (fetched via API and cached).
- Handles text and image inputs (with size validation).
- Supports streaming and non-streaming responses.
- Easy to configure with your Anthropic API key.

---

## Notes

- Images must be under 5MB each and total under 100MB for a request.
- The model name should be prefixed with `"anthropic."` (e.g., `"anthropic.claude-3-opus-20240229"`).
- Errors during requests are returned as strings.
- The list of models is cached for 10 minutes by default.  This can be changed by setting the `ANTHROPIC_MODEL_CACHE_TTL` environment variable.

---

## License

MIT License

---

## Authors

- justinh-rahb ([GitHub](https://github.com/justinh-rahb))
- christian-taillon
- jfbloom22 ([GitHub](https://github.com/jfbloom22))