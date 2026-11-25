# Vendored: Depth Anything 3

The 3D foundation model behind every pass of Xtream (DA3-GIANT-1.1 for the windows,
covisibility edges, loops and map passes; DA3METRIC-LARGE for the monocular metric prior).

- **Upstream:** https://github.com/ByteDance-Seed/Depth-Anything-3
- **Commit:** 3d835ec
- **License:** Apache-2.0 (see `LICENSE`)

Plain copy of the package under `src/`, unmodified. **Trimmed**: `app/`, `bench/`,
`cli.py` and `services/` were removed (not imported by the inference API). Xtream puts
`src/` on `sys.path`; weights download from the Hugging Face hub on first use.
