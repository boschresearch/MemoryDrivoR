# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Bench2Drive route XML containing selected towns.")
    parser.add_argument("--input", type=Path, required=True, help="Source Bench2Drive route XML.")
    parser.add_argument("--output", type=Path, required=True, help="Destination route XML.")
    parser.add_argument(
        "--town",
        action="append",
        required=True,
        help="Town name to keep, for example Town11. Repeat to keep multiple towns.",
    )
    return parser.parse_args()


def filter_routes_by_town(input_path: Path, output_path: Path, towns: Sequence[str]) -> int:
    town_set = set(towns)
    source_tree = ET.parse(input_path)
    source_root = source_tree.getroot()
    output_root = ET.Element(source_root.tag, source_root.attrib)

    matched = 0
    for route in source_root.findall("route"):
        if route.attrib.get("town") in town_set:
            output_root.append(copy.deepcopy(route))
            matched += 1

    if matched == 0:
        towns_label = ", ".join(sorted(town_set))
        raise ValueError(f"No routes for town(s) {towns_label} found in {input_path}.")

    output_tree = ET.ElementTree(output_root)
    if hasattr(ET, "indent"):
        ET.indent(output_tree, space="   ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return matched


def main() -> None:
    args = _parse_args()
    matched = filter_routes_by_town(args.input, args.output, args.town)
    towns_label = ", ".join(args.town)
    print(f"Wrote {matched} route(s) for {towns_label} to {args.output}")


if __name__ == "__main__":
    main()
