# CLI Reference

AutoMR exposes a simple command-line interface for quick inspection and discovery of bundled documentation.

Available options

- `--version` : Print the installed AutoMR package version.
- `--info`    : Print a short feature summary and the framework version.
- `--docs`    : List documentation files included in the package (shows files under the `docs/` directory).

Examples

```bash
automr --version
automr --info
automr --docs
```

Notes

- The `--docs` option enumerates files found under the package `docs/` folder as shipped with AutoMR.
- Use `automr --info` when you need a quick summary of features and supported backends.
