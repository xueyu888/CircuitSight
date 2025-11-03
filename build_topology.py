import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

# Hard-coded interpretation of the four panels in the provided reference image
# (see reference.txt for the bundled artwork; Base64-decode it to restore the PNG)
circuits = [
    {
        "name": "Panel 1 - Surge protection with indicator and arrestor",
        "nodes": [
            {"id": "bus", "label": "Incoming line"},
            {"id": "ground", "label": "Ground"}
        ],
        "components": [
            {
                "id": "LP1",
                "type": "indicator_lamp",
                "connections": ["bus", "ground"],
                "note": "Lamp with one side tied to the line and the other to ground"
            },
            {
                "id": "D1",
                "type": "surge_diode",
                "connections": ["bus", "ground"],
                "note": "Diode symbol to ground, interpreted as a discharge path"
            },
            {
                "id": "SPD1",
                "type": "surge_module",
                "connections": ["bus", "ground"],
                "note": "Box with a downward triangle, treated as a surge protection cartridge"
            }
        ],
        "layout": {
            "bus": (0.0, 1.0),
            "ground": (0.0, -1.0)
        }
    },
    {
        "name": "Panel 2 - Lamp with single discharge diode",
        "nodes": [
            {"id": "bus", "label": "Incoming line"},
            {"id": "ground", "label": "Ground"}
        ],
        "components": [
            {
                "id": "LP1",
                "type": "indicator_lamp",
                "connections": ["bus", "ground"],
                "note": "Lamp monitoring the line"
            },
            {
                "id": "D1",
                "type": "surge_diode",
                "connections": ["bus", "ground"],
                "note": "Single line-to-ground discharge diode"
            }
        ],
        "layout": {
            "bus": (0.0, 1.0),
            "ground": (0.0, -1.0)
        }
    },
    {
        "name": "Panel 3 - Lamp with earth switch and discharge diode",
        "nodes": [
            {"id": "bus", "label": "Incoming line"},
            {"id": "ground", "label": "Ground"},
            {"id": "earth_switch", "label": "Earth switch contact"}
        ],
        "components": [
            {
                "id": "S1",
                "type": "earth_switch",
                "connections": ["bus", "earth_switch"],
                "note": "Open earthing switch tied to ground"
            },
            {
                "id": "LP1",
                "type": "indicator_lamp",
                "connections": ["bus", "ground"],
                "note": "Lamp connected from the line to ground"
            },
            {
                "id": "D1",
                "type": "surge_diode",
                "connections": ["bus", "ground"],
                "note": "Line-to-ground discharge diode"
            }
        ],
        "layout": {
            "bus": (0.0, 1.0),
            "ground": (0.5, -1.0),
            "earth_switch": (-0.5, -1.0)
        }
    },
    {
        "name": "Panel 4 - Measurement loop with diode and disconnect",
        "nodes": [
            {"id": "bus", "label": "Incoming line"},
            {"id": "ground", "label": "Ground"},
            {"id": "loop_after_diode", "label": "After diode"},
            {"id": "loop_after_link", "label": "After disconnect link"},
            {"id": "loop_after_v1", "label": "After voltmeter V1"},
            {"id": "loop_after_v2", "label": "After voltmeter V2"}
        ],
        "components": [
            {
                "id": "LP1",
                "type": "indicator_lamp",
                "connections": ["bus", "ground"],
                "note": "Lamp with one side tied to ground"
            },
            {
                "id": "D1",
                "type": "diode",
                "connections": ["bus", "loop_after_diode"],
                "note": "Diode feeding the measurement loop"
            },
            {
                "id": "X1",
                "type": "disconnect_link",
                "connections": ["loop_after_diode", "loop_after_link"],
                "note": "Circle with slash interpreted as a removable link"
            },
            {
                "id": "V1",
                "type": "voltmeter",
                "connections": ["loop_after_link", "loop_after_v1"],
                "note": "Upper voltmeter in the loop"
            },
            {
                "id": "V2",
                "type": "voltmeter",
                "connections": ["loop_after_v1", "loop_after_v2"],
                "note": "Lower voltmeter in the loop"
            },
            {
                "id": "R1",
                "type": "resistor",
                "connections": ["loop_after_v2", "bus"],
                "note": "Rectangular element interpreted as a series resistor closing the loop"
            }
        ],
        "layout": {
            "bus": (0.0, 1.0),
            "ground": (-0.8, -0.8),
            "loop_after_diode": (0.0, 0.2),
            "loop_after_link": (0.5, 0.0),
            "loop_after_v1": (0.5, -0.4),
            "loop_after_v2": (0.2, -0.6)
        }
    }
]

def render_topology_figure(panels):
    """Render the interpreted topology diagrams and return the PNG bytes."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, circuit in zip(axes, panels):
        graph = nx.Graph()
        pos = circuit["layout"].copy()
        for node in circuit["nodes"]:
            node_id = node["id"]
            label = node.get("label", node_id)
            graph.add_node(node_id, label=label)
            if node_id not in pos:
                pos[node_id] = (0.0, 0.0)

        for component in circuit["components"]:
            connections = component["connections"]
            if len(connections) < 2:
                continue
            for start, end in zip(connections, connections[1:]):
                edge_label = f"{component['id']} ({component['type']})"
                graph.add_edge(start, end, label=edge_label)

        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color="#1f77b4", node_size=900)
        nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=8, font_color="white")

        edge_labels = nx.get_edge_attributes(graph, "label")
        nx.draw_networkx_edges(graph, pos=pos, ax=ax, arrows=False)
        nx.draw_networkx_edge_labels(graph, pos=pos, ax=ax, edge_labels=edge_labels, font_size=6)

        ax.set_title(circuit["name"], fontsize=10)
        ax.axis("off")

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Export the interpreted circuit topology data and diagrams.")
    parser.add_argument(
        "--emit-png",
        action="store_true",
        help="Also emit a binary PNG alongside the text-friendly Base64 output.",
    )
    args = parser.parse_args()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "circuit_topology.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"reference_image": "reference.png", "circuits": circuits}, fh, indent=2, ensure_ascii=False)

    png_bytes = render_topology_figure(circuits)
    b64_payload = base64.b64encode(png_bytes).decode("ascii")
    text_path = output_dir / "circuit_topology.txt"
    text_path.write_text(b64_payload, encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {text_path} (Base64-encoded PNG)")

    if args.emit_png:
        plot_path = output_dir / "circuit_topology.png"
        plot_path.write_bytes(png_bytes)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
