# Contributing to SmartGallery

Thank you for considering contributing to SmartGallery DAM! 

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Running the tests

The suite is model-free: it downloads no weights and needs no GPU.

```
uv sync
uv run pytest tests/
```

`just test` runs the same suite through the dev venv, and `just audit` runs
the structural checks alone in a few seconds.

## Reporting Issues

Please use the GitHub issue tracker to report bugs or request features.