# Setting Up Project Sync for Other Projects

This guide explains how to use the `sync.sh` script in other projects.

## Quick Setup

1. **Copy the script to your new project**:
   ```bash
   cp /path/to/BigDeep/sync.sh /path/to/your-project/
   chmod +x /path/to/your-project/sync.sh
   ```

2. **Create a `.sync.conf` file** in your project root:
   ```bash
   cp /path/to/BigDeep/.sync.conf.example /path/to/your-project/.sync.conf
   ```

3. **Edit `.sync.conf`** to match your project:
   - Update the `[project]` section with your project-specific settings
   - Add your remote host presets
   - Remove or modify the `uv_projects` and `path_fix` settings as needed

4. **Run the sync**:
   ```bash
   ./sync.sh --preset your-preset-name
   ```

## Configuration Examples

### Simple Project (No UV Dependencies)
If your project doesn't use uv or doesn't need path fixes, you can omit the `[project]` section entirely:

```ini
# .sync.conf
[laptop]
host=user@laptop
dir=/home/user/projects/MyApp

[server]
host=deploy@prod.example.com
dir=/opt/MyApp
key=~/.ssh/prod_key
```

### Python Project with UV
For Python projects using uv:

```ini
[project]
uv_projects=backend frontend
# No path fixes needed

[dev]
host=dev.internal
dir=/workspace/MyProject
```

### Monorepo with Path Fixes
For complex projects with relative dependencies:

```ini
[project]
uv_projects=service-a service-b shared-lib
path_fix=sed -i 's|../shared|$REMOTE_DIR/shared|g' $REMOTE_DIR/service-a/pyproject.toml
path_fix=sed -i 's|../shared|$REMOTE_DIR/shared|g' $REMOTE_DIR/service-b/pyproject.toml

[remote]
host=build-server
dir=/builds/MyMonorepo
```

## Features

- **Auto-detection**: The script automatically detects the project name from the directory
- **Live sync**: Watches for file changes and syncs automatically (use `--no-watch` to disable)
- **Dependency management**: Automatically installs rsync, uv, and tmux on the remote if needed
- **Flexible configuration**: Support for multiple remote presets
- **Path fixing**: Automatically fix relative paths that break when syncing

## Command Options

```bash
./sync.sh --preset desktop              # Use a preset
./sync.sh --preset remote --no-watch    # One-time sync
./sync.sh --preset remote --skip-deps   # Skip dependency installation
./sync.sh --list-presets                # List available presets
./sync.sh --help                        # Show all options
```

## Global Installation (Optional)

To use the script globally across all projects:

1. Copy to your bin directory:
   ```bash
   cp sync.sh ~/bin/project-sync
   chmod +x ~/bin/project-sync
   ```

2. Each project still needs its own `.sync.conf` file

3. Run from any project directory:
   ```bash
   cd /path/to/your-project
   project-sync --preset your-preset
   ```

## Tips

- Add `.sync.conf` to `.gitignore` if it contains sensitive information
- Use `.sync.conf.example` as a template for team members
- The script excludes common directories like `.git`, `__pycache__`, `.venv`, etc.
- You can customize the exclude list by editing the `EXCLUDES` array in `sync.sh`
