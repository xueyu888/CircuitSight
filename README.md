# CircuitSight

This repository extracts and visualizes the interpreted topology of the supplied circuit drawing.

## Reference image

The original reference drawing is stored as a Base64 text payload in `reference.txt`. Decode it back to
a PNG when you need to view it locally:

```bash
base64 -d reference.txt > reference.png
```

You can then open `reference.png` with any image viewer.

## Topology exports

Running `python build_topology.py` regenerates both the interpreted connectivity JSON and a
Base64-encoded version of the comparison plot:

```bash
python build_topology.py
# output/circuit_topology.json
# output/circuit_topology.txt  # Base64 PNG payload
```

To restore the image locally without committing binary artefacts, decode the text payload or
ask the script to emit the PNG on demand:

```bash
base64 -d output/circuit_topology.txt > output/circuit_topology.png
# or generate it directly
python build_topology.py --emit-png
```
