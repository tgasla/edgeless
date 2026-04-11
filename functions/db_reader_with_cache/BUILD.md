# Building db_reader_with_cache

This function is built for **ARM native** (Raspberry Pi) and requires the Edgeless CLI for proper placement.

## Build Commands

### ARM Native (for RPi edgexpert-95b2 node)
```bash
cd /home/theo/edgeless
target/debug/edgeless_cli function build functions/db_reader_with_cache/function.json --architecture arm
```

### WASM (for local testing/dev)
```bash
cargo build --manifest-path functions/db_reader_with_cache/Cargo.toml --target wasm32-unknown-unknown --release
```

## Why edgeless_cli instead of cargo?

`cargo build` outputs to `target/aarch64-unknown-linux-gnu/release/libdb_reader_with_cache.so`

The workflow JSON references `functions/db_reader_with_cache/db_reader_with_cache_aarch.so`

`edgeless_cli function build` automatically copies the output to the correct location in the function directory.

Using cargo directly means the .so stays in the target/ subdirectory and is NOT picked up by the workflow orchestrator.

## Important: Binary Placement

The .so files do NOT need to be on the target node manually. The node that submits the workflow (e.g., MSI) maintains a repository of all function binaries. When the workflow starts, the orchestrator pushes the correct binary to each target node based on annotations.

So: build on any node that has the binaries, and the orchestrator handles distribution.

## After Building

1. The .so file in `functions/db_reader_with_cache/` is automatically picked up by the workflow orchestrator
2. Restart the workflow from the controller node
3. Clear stale Redis cache: `redis-cli DEL image_history`
