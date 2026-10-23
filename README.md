# SF-Swin-UNETR — LecTrace presentation

A ~10-minute step-through lecture built with [LecTrace](https://praisegee.github.io/lectrace/).
It walks through the core idea of the Spatial-Frequency Gate (SF-Gate): why a
frequency (FFT) branch catches the subtle GGO texture that a spatial
(channel-attention) branch averages away, then shows the real benchmark result.

## Requirements

- Python 3.11 or 3.12
- `lectrace`, `numpy`, `matplotlib`

## Install

```bash
pip install lectrace numpy matplotlib
```

## Run

```bash
# live server with auto-reload (recommended while editing)
lectrace serve
```

Then open **http://localhost:7000**. Use the left/right arrow keys to step
through the code line by line; the panel on the right shows the variables marked
with `# @inspect`.

Other commands:

```bash
lectrace build                      # static site in _site/
lectrace run 01_sf_swin_unetr.py    # run one file and print trace info
lectrace init                       # GitHub Actions workflow for deployment
```

### If `lectrace` is not recognized on Windows

The Microsoft Store Python installs console scripts into a directory that is
often not on `PATH`. Find the `lectrace.exe` and call it by full path:

```powershell
# locate it:
pip show lectrace        # -> note the "Location" (site-packages) path

# run it (replace with your own path):
& "C:\Users\yansh\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\lectrace.exe" serve
```

Or invoke the CLI entry point through `python`:

```powershell
python -c "from lectrace.cli import main; main()" serve
```

Or add the Scripts directory to `PATH` permanently (then restart the terminal):

```powershell
$p = "C:\Users\yansh\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts"
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path","User") + ";" + $p, "User")
```

## Files

| File | What it is |
|---|---|
| `01_sf_swin_unetr.py` | The lecture: intro, 5 execution states, conclusion |
| `_helpers.py`         | Plot helpers (underscore keeps it out of the trace) |
| `lectrace.toml`       | Config (title) |
| `assets/`             | Architecture / CT example / qualitative images |

## The five execution states

1. **Input** — a lesion profile reduced to one dimension (bump + faint ripple).
2. **Spatial branch** — channel attention's global average cancels the ripple.
3. **Frequency branch** — the FFT magnitude exposes the ripple as a peak.
4. **Gated fusion** — a learned gate combines the two branches.
5. **Real result** — Dice / HD95 table + a qualitative slice + clinical meaning.
