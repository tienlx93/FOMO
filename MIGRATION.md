# Migration Guide: From pip to UV

This document explains how to migrate the FOMO project from traditional pip-based dependency management to UV.

## What Changed

### Files Added
- `pyproject.toml` - Project configuration and dependencies
- `.python-version` - Python version specification for UV

### Files Modified
- `README.md` - Updated setup and usage instructions

### Files to Remove (Optional)
- `requirements.txt` - No longer needed (kept for reference)

## Quick Start with UV

1. **Install UV (if not already installed):**
   
   Follow the [official UV installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your operating system.

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Run the application:**
   ```bash
   uv run streamlit run main.py
   ```

## Key Benefits of UV

- **Faster dependency resolution** - UV is significantly faster than pip
- **Reproducible builds** - `uv.lock` ensures exact versions across environments
- **Better dependency management** - Handles conflicts more intelligently
- **Built-in virtual environment management** - No need to manually create venvs
- **Modern Python packaging** - Uses pyproject.toml standard

## UV Commands Cheat Sheet

| Task | UV Command | Old pip Command |
|------|------------|-----------------|
| Install dependencies | `uv sync` | `pip install -r requirements.txt` |
| Add new dependency | `uv add package-name` | `pip install package-name` |
| Remove dependency | `uv remove package-name` | `pip uninstall package-name` |
| Run script | `uv run python script.py` | `python script.py` |
| Update dependencies | `uv lock --upgrade` | `pip install --upgrade -r requirements.txt` |
| Show installed packages | `uv pip list` | `pip list` |

## Development Workflow

1. **Adding new dependencies:**
   ```bash
   uv add streamlit  # Add to main dependencies
   uv add --dev pytest  # Add to development dependencies
   ```

2. **Updating dependencies:**
   ```bash
   uv lock --upgrade  # Update lock file with latest versions
   uv sync  # Install updated dependencies
   ```

3. **Running tests:**
   ```bash
   uv run python test_app.py
   ```

## Project Structure

```
FOMO/
├── pyproject.toml      # Project config & dependencies
├── .python-version     # Python version for UV
├── uv.lock            # Lock file (auto-generated)
├── .venv/             # Virtual environment (auto-managed)
├── main.py            # Application files
├── backend_developer.py
├── test_app.py
└── data/
```

## Troubleshooting

- **UV not found**: Install following the [official UV installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **Permission issues**: Make sure UV installation directory is in your PATH
- **Dependency conflicts**: UV will show clear error messages and suggestions
- **Python version issues**: Check `.python-version` file matches your system

For more information, visit the [UV documentation](https://docs.astral.sh/uv/).