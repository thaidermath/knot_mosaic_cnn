# This is just adding the crossings based on the PD string using tiles 10,11 for crossings. And tile 8 and 9 for dual arcs.

import os
import random
import re
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image


PORTS = "abcd"
DIRS = ("N", "E", "S", "W")
DIR_TO_DELTA = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
CCW_ORDER = ["N", "W", "S", "E"]
DIR_INDEX = {d: i for i, d in enumerate(CCW_ORDER)}
PORT_INDEX = {p: i for i, p in enumerate(PORTS)}
SINGLE_ARC_TILES = {
    frozenset({"W", "S"}): 2,
    frozenset({"S", "E"}): 3,
    frozenset({"E", "N"}): 4,
    frozenset({"N", "W"}): 5,
    frozenset({"W", "E"}): 6,
    frozenset({"N", "S"}): 7,
}
DUAL_ARC_TILES = {
    frozenset({frozenset({"W", "S"}), frozenset({"E", "N"})}): 8,
    frozenset({frozenset({"N", "W"}), frozenset({"S", "E"})}): 9,
}
DUAL_RULES = [
    {
        "tile": 8,
        "pairs": (frozenset({"W", "S"}), frozenset({"E", "N"})),
        "exit": {"W": "S", "S": "W", "E": "N", "N": "E"},
    },
    {
        "tile": 9,
        "pairs": (frozenset({"N", "W"}), frozenset({"S", "E"})),
        "exit": {"N": "W", "W": "N", "S": "E", "E": "S"},
    },
]
TURN_LEFT = {"N": "W", "W": "S", "S": "E", "E": "N"}
TURN_RIGHT = {"N": "E", "E": "S", "S": "W", "W": "N"}


def _build_tile_port_map() -> Dict[int, List[Tuple[str, ...]]]:
    mapping: Dict[int, List[Tuple[str, ...]]] = {}
    for ports, tile_id in SINGLE_ARC_TILES.items():
        mapping.setdefault(tile_id, []).append(tuple(sorted(ports)))
    for combos, tile_id in DUAL_ARC_TILES.items():
        mapping.setdefault(tile_id, []).extend(tuple(sorted(p)) for p in combos)
    crossing_ports = [("N", "S"), ("E", "W")]
    for tile_id in (10, 11):
        mapping.setdefault(tile_id, []).extend(crossing_ports)
    return mapping


TILE_PORT_MAP = _build_tile_port_map()


class RoutingConflict(Exception):
    pass


def parse_pd_string(pd_str: str) -> List[Tuple[int, int, int, int]]:
    quads = re.findall(r"X\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", pd_str)
    if not quads:
        raise ValueError("No X[...] tuples found in PD string")
    pd = [tuple(map(int, quad)) for quad in quads]
    counts = defaultdict(int)
    for quad in pd:
        for label in quad:
            counts[label] += 1
    bad = [lbl for lbl, freq in counts.items() if freq != 2]
    if bad:
        raise ValueError(f"Arc labels must appear exactly twice: {bad}")
    return pd


def build_label_lookup(pd: List[Tuple[int, int, int, int]]):
    lookup: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for idx, quad in enumerate(pd):
        for port, label in zip(PORTS, quad):
            lookup[label].append((idx, port))
    return lookup


def trim_tile_matrix(tile_mat: List[List[int]]) -> List[List[int]]:
    coords = [
        (r, c)
        for r in range(len(tile_mat))
        for c in range(len(tile_mat[0]))
        if tile_mat[r][c] != 0
    ]
    if not coords:
        return tile_mat
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)
    return [row[min_c : max_c + 1] for row in tile_mat[min_r : max_r + 1]]


def pad_tile_matrix(tile_mat: List[List[int]], pad: int = 1) -> List[List[int]]:
    if pad <= 0 or not tile_mat:
        return tile_mat
    height = len(tile_mat)
    width = len(tile_mat[0])
    padded = [[0] * (width + 2 * pad) for _ in range(height + 2 * pad)]
    for r in range(height):
        for c in range(width):
            padded[r + pad][c + pad] = tile_mat[r][c]
    return padded


def validate_tile_matrix(tile_mat: List[List[int]], filler: int = 1) -> None:
    if not tile_mat:
        return
    height = len(tile_mat)
    width = len(tile_mat[0])
    for r in range(height):
        for c in range(width):
            tile = tile_mat[r][c]
            if tile == 0 or tile == filler:
                continue
            port_groups = TILE_PORT_MAP.get(tile)
            if not port_groups:
                continue
            for group in port_groups:
                for port in group:
                    dr, dc = DIR_TO_DELTA[port]
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        raise ValueError(
                            f"Open strand at ({r}, {c}) exits grid via {port}"
                        )
                    neighbor = tile_mat[nr][nc]
                    if neighbor == 0 or neighbor == filler:
                        raise ValueError(
                            f"Open strand at ({r}, {c}) via {port} enters empty cell"
                        )
                    neighbor_groups = TILE_PORT_MAP.get(neighbor, [])
                    if not any(OPPOSITE[port] in ng for ng in neighbor_groups):
                        raise ValueError(
                            f"Port mismatch between ({r}, {c}) {port} and ({nr}, {nc})"
                        )


def fill_empty_tiles(tile_mat: List[List[int]], filler: int = 1) -> List[List[int]]:
    return [[cell if cell != 0 else filler for cell in row] for row in tile_mat]


def decide_crossing_tile(port: str, direction: str) -> int:
    is_vertical = direction in ("N", "S")
    if is_vertical:
        return 10 if port in ("a", "c") else 11
    return 11 if port in ("a", "c") else 10


def port_mapping_from_anchor(anchor_port: str, anchor_side: str) -> Dict[str, str]:
    if anchor_port not in PORT_INDEX:
        raise ValueError(f"Invalid anchor port {anchor_port}")
    if anchor_side not in CCW_ORDER:
        raise ValueError(f"Invalid anchor side {anchor_side}")
    start_dir_idx = DIR_INDEX[anchor_side]
    ordered_ports = [PORTS[(PORT_INDEX[anchor_port] + i) % 4] for i in range(4)]
    mapping = {}
    for offset, port in enumerate(ordered_ports):
        direction = CCW_ORDER[(start_dir_idx + offset) % 4]
        mapping[port] = direction
    return mapping


class MosaicBuilder:
    def __init__(self, pd: List[Tuple[int, int, int, int]]):
        self.pd = pd
        self.label_occ = build_label_lookup(pd)
        size = max(32, len(pd) * 6 + 16)
        self.height = size
        self.width = size
        self.center = (size // 2, size // 2)
        self.tile_mat = [[0] * size for _ in range(size)]
        self.crossing_cells: Set[Tuple[int, int]] = set()
        self.crossing_positions: Dict[int, Tuple[int, int]] = {}
        self.orientation_map: Dict[int, Dict[str, str]] = {}
        self.placed_crossings: Set[int] = set()
        self.open_ends_by_label: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        self.open_ends_by_cell: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
        self.occupied_cells: Set[Tuple[int, int]] = set()
        self.cell_ports: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(dict)

    def _ensure_in_bounds(self, r: int, c: int):
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError("Grid too small for placement; increase base size")

    def _labels_share_crossing(self, a: int, b: int) -> bool:
        if a == b:
            return True
        for quad in self.pd:
            labels = set(quad)
            if a in labels and b in labels:
                return True
        return False

    def _register_open_end(self, stub: Dict[str, object]):
        label = stub["label"]  # type: ignore[index]
        cell = stub["cell"]  # type: ignore[index]
        self.open_ends_by_label[label].append(stub)
        self.open_ends_by_cell[cell].append(stub)
        self._maybe_place_dual_tile(cell)

    def _remove_open_end(self, stub: Dict[str, object]):
        label = stub["label"]  # type: ignore[index]
        cell = stub["cell"]  # type: ignore[index]
        if label in self.open_ends_by_label:
            lst = self.open_ends_by_label[label]
            if stub in lst:
                lst.remove(stub)
            if not lst:
                del self.open_ends_by_label[label]
        if cell in self.open_ends_by_cell:
            clst = self.open_ends_by_cell[cell]
            if stub in clst:
                clst.remove(stub)
            if not clst:
                del self.open_ends_by_cell[cell]

    def _clear_open_ends_at_cell(self, cell: Tuple[int, int]):
        entries = self.open_ends_by_cell.get(cell)
        if not entries:
            return
        for stub in list(entries):
            self._remove_open_end(stub)

    def _set_cell_port(self, cell: Tuple[int, int], direction: str, label: int):
        ports = self.cell_ports[cell]
        existing = ports.get(direction)
        if existing is not None and existing != label:
            raise ValueError(
                f"Port conflict at {cell} {direction}: {existing} vs {label}"
            )
        ports[direction] = label

    def _update_tile_from_ports(self, cell: Tuple[int, int]):
        if cell in self.crossing_cells:
            return
        ports = self.cell_ports.get(cell)
        if not ports:
            return
        used = [side for side in DIRS if ports.get(side) is not None]
        if not used:
            return
        r, c = cell
        if len(used) == 2:
            labels = {ports[side] for side in used}
            if len(labels) != 1:
                raise ValueError(f"Mismatched labels in cell {cell}: {labels}")
            tile = SINGLE_ARC_TILES.get(frozenset(used))
            if tile is None:
                raise ValueError(f"Unsupported connection at {cell}: {used}")
            self.tile_mat[r][c] = tile
            self.occupied_cells.add(cell)
        elif len(used) == 4:
            label_groups: Dict[int, List[str]] = defaultdict(list)
            for side in DIRS:
                lbl = ports.get(side)
                if lbl is not None:
                    label_groups[lbl].append(side)
            if len(label_groups) != 2:
                raise ValueError(f"Dual arcs at {cell} must involve two labels")
            shapes = frozenset(frozenset(v) for v in label_groups.values())
            tile = DUAL_ARC_TILES.get(shapes)
            if tile is None:
                raise ValueError(f"Unsupported dual connection at {cell}: {label_groups}")
            self.tile_mat[r][c] = tile
            self.occupied_cells.add(cell)

    def _cell_allows_port(
        self,
        cell: Tuple[int, int],
        port: str,
        label: int,
        endpoints: Set[Tuple[int, int]],
    ) -> bool:
        if cell in self.crossing_cells:
            return False
        if cell not in endpoints:
            for stub in self.open_ends_by_cell.get(cell, []):
                if stub["label"] != label:  # type: ignore[index]
                    return False
        ports = dict(self.cell_ports.get(cell, {}))
        existing = ports.get(port)
        if existing is not None and existing != label:
            return False
        ports[port] = label
        label_groups: Dict[int, List[str]] = defaultdict(list)
        for side, lbl in ports.items():
            label_groups[lbl].append(side)
            if len(label_groups[lbl]) > 2:
                return False
        if len(label_groups) > 2:
            return False
        if len(ports) > 4:
            return False
        if len(label_groups) == 2 and len(ports) == 4:
            shapes = frozenset(frozenset(sides) for sides in label_groups.values())
            if shapes not in DUAL_ARC_TILES:
                return False
        return True

    def _ordered_dirs(self, heading: str, prefer_left: bool) -> List[str]:
        if prefer_left:
            return [
                TURN_LEFT[heading],
                heading,
                TURN_RIGHT[heading],
                OPPOSITE[heading],
            ]
        return [
            TURN_RIGHT[heading],
            heading,
            TURN_LEFT[heading],
            OPPOSITE[heading],
        ]

    def _can_travel(
        self,
        cell: Tuple[int, int],
        direction: str,
        label: int,
        endpoints: Set[Tuple[int, int]],
    ) -> bool:
        dr, dc = DIR_TO_DELTA[direction]
        nr = cell[0] + dr
        nc = cell[1] + dc
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return False
        next_cell = (nr, nc)
        if next_cell in self.crossing_cells:
            return False
        if not self._cell_allows_port(cell, direction, label, endpoints):
            return False
        if not self._cell_allows_port(next_cell, OPPOSITE[direction], label, endpoints):
            return False
        return True

    def _reconstruct_moves(
        self,
        parents: Dict[Tuple[int, int, str], Optional[Tuple[int, int, str]]],
        moves: Dict[Tuple[int, int, str], Optional[str]],
        state: Tuple[int, int, str],
    ) -> List[str]:
        seq: List[str] = []
        cur = state
        while parents[cur] is not None:
            move = moves[cur]
            if move is None:
                break
            seq.append(move)
            cur = parents[cur]  # type: ignore[index]
        seq.reverse()
        return seq

    def _wall_follow_path(
        self,
        start_stub: Dict[str, object],
        goal_stub: Dict[str, object],
        label: int,
        prefer_left: bool,
    ) -> Optional[List[str]]:
        start_cell = start_stub["cell"]  # type: ignore[index]
        goal_cell = goal_stub["cell"]  # type: ignore[index]
        if start_cell == goal_cell:
            return []
        start_heading = OPPOSITE[start_stub["port"]]  # type: ignore[index]
        queue = deque()
        start_state = (start_cell[0], start_cell[1], start_heading)
        queue.append(start_state)
        parents: Dict[Tuple[int, int, str], Optional[Tuple[int, int, str]]] = {
            start_state: None
        }
        moves: Dict[Tuple[int, int, str], Optional[str]] = {start_state: None}
        endpoints = {start_cell, goal_cell}
        visited = {start_state}

        while queue:
            r, c, heading = queue.popleft()
            cell = (r, c)
            if cell == goal_cell and (r, c, heading) != start_state:
                entry_port = OPPOSITE[heading]
                pair = frozenset({entry_port, goal_stub["port"]})  # type: ignore[index]
                if (
                    entry_port != goal_stub["port"]  # type: ignore[index]
                    and pair in SINGLE_ARC_TILES
                    and self._cell_allows_port(goal_cell, goal_stub["port"], label, endpoints)  # type: ignore[index]
                ):
                    return self._reconstruct_moves(parents, moves, (r, c, heading))
            for direction in self._ordered_dirs(heading, prefer_left):
                if not self._can_travel(cell, direction, label, endpoints):
                    continue
                nr = r + DIR_TO_DELTA[direction][0]
                nc = c + DIR_TO_DELTA[direction][1]
                next_state = (nr, nc, direction)
                if next_state in visited:
                    continue
                visited.add(next_state)
                parents[next_state] = (r, c, heading)
                moves[next_state] = direction
                queue.append(next_state)
        return None

    def _wall_paths_for_pair(
        self,
        stub_a: Dict[str, object],
        stub_b: Dict[str, object],
        label: int,
        forbidden: Optional[Set[Tuple[str, ...]]] = None,
    ) -> List[Tuple[bool, List[str]]]:
        results: List[Tuple[bool, List[str]]] = []
        forbidden = forbidden or set()
        if stub_a["cell"] == stub_b["cell"]:  # type: ignore[index]
            results.append((True, []))
            return results
        for prefer_left in (True, False):
            path = self._wall_follow_path(stub_a, stub_b, label, prefer_left)
            if path is None:
                continue
            path_key = tuple(path)
            if path_key in forbidden:
                continue
            results.append((prefer_left, path))
        return results

    def _path_metrics(
        self,
        stub_a: Dict[str, object],
        moves: List[str],
    ) -> Tuple[int, int]:
        """Return (fresh_cells, reused_cells) touched when applying moves."""
        fresh = 0
        reused = 0
        cells: List[Tuple[int, int]] = []
        cell = stub_a["cell"]  # type: ignore[index]
        cells.append(cell)
        for direction in moves:
            dr, dc = DIR_TO_DELTA[direction]
            cell = (cell[0] + dr, cell[1] + dc)
            cells.append(cell)
        for r, c in cells:
            if (r, c) in self.crossing_cells:
                continue
            if self.tile_mat[r][c] == 0:
                fresh += 1
            else:
                reused += 1
        return fresh, reused

    def _apply_same_cell_connection(
        self,
        stub_a: Dict[str, object],
        stub_b: Dict[str, object],
        label: int,
    ):
        cell = stub_a["cell"]  # type: ignore[index]
        port_a = stub_a["port"]  # type: ignore[index]
        port_b = stub_b["port"]  # type: ignore[index]
        pair = frozenset({port_a, port_b})
        if port_a == port_b or pair not in SINGLE_ARC_TILES:
            raise ValueError(f"Cannot join label {label} stubs occupying {cell}")
        self._set_cell_port(cell, port_a, label)
        self._set_cell_port(cell, port_b, label)
        self._update_tile_from_ports(cell)

    def _apply_path(
        self,
        stub_a: Dict[str, object],
        stub_b: Dict[str, object],
        label: int,
        moves: List[str],
    ):
        if not moves:
            self._apply_same_cell_connection(stub_a, stub_b, label)
            return
        touched_order: List[Tuple[int, int]] = []
        saved_ports: Dict[Tuple[int, int], Dict[str, int]] = {}
        saved_tiles: Dict[Tuple[int, int], int] = {}

        def snapshot(cell: Tuple[int, int]):
            if cell in saved_ports:
                return
            saved_ports[cell] = dict(self.cell_ports.get(cell, {}))
            r, c = cell
            saved_tiles[cell] = self.tile_mat[r][c]
            touched_order.append(cell)

        cell = stub_a["cell"]  # type: ignore[index]
        entry_port = stub_a["port"]  # type: ignore[index]
        try:
            for direction in moves:
                snapshot(cell)
                self._set_cell_port(cell, entry_port, label)
                self._set_cell_port(cell, direction, label)
                self._update_tile_from_ports(cell)
                dr, dc = DIR_TO_DELTA[direction]
                cell = (cell[0] + dr, cell[1] + dc)
                entry_port = OPPOSITE[direction]
            snapshot(cell)
            self._set_cell_port(cell, entry_port, label)
            self._set_cell_port(cell, stub_b["port"], label)  # type: ignore[index]
            self._update_tile_from_ports(cell)
        except Exception as exc:  # noqa: BLE001
            for revert_cell in reversed(touched_order):
                prev_ports = saved_ports[revert_cell]
                if prev_ports:
                    self.cell_ports[revert_cell] = dict(prev_ports)
                else:
                    self.cell_ports.pop(revert_cell, None)
                r, c = revert_cell
                prev_tile = saved_tiles[revert_cell]
                self.tile_mat[r][c] = prev_tile
                if prev_tile == 0:
                    self.occupied_cells.discard(revert_cell)
                else:
                    self.occupied_cells.add(revert_cell)
            raise RoutingConflict(str(exc)) from exc

    def _verify_open_end_pairs(self):
        bad = {
            label: len(stubs)
            for label, stubs in self.open_ends_by_label.items()
            if len(stubs) not in (0, 2)
        }
        if bad:
            details = ", ".join(
                f"label {lbl}: {count} stubs" for lbl, count in sorted(bad.items())
            )
            raise ValueError(f"Open-end bookkeeping lost track of labels ({details})")

    def route_open_ends(self):
        if not self.open_ends_by_label:
            return
        self._verify_open_end_pairs()
        invalid_paths: Dict[int, Set[Tuple[str, ...]]] = defaultdict(set)
        while self.open_ends_by_label:
            candidates: List[
                Tuple[int, int, int, int, Dict[str, object], Dict[str, object], List[str]]
            ] = []
            for label, stubs in list(self.open_ends_by_label.items()):
                if len(stubs) != 2:
                    raise ValueError(
                        f"Label {label} expected 2 open ends, found {len(stubs)}"
                    )
                stub_a, stub_b = stubs
                disallow = invalid_paths.get(label)
                for prefer_left, moves in self._wall_paths_for_pair(
                    stub_a, stub_b, label, disallow
                ):
                    turn_priority = 0 if prefer_left else 1
                    fresh, reused = self._path_metrics(stub_a, moves)
                    candidates.append(
                        (fresh, len(moves), -reused, label, turn_priority, stub_a, stub_b, moves)
                    )
            if not candidates:
                raise ValueError("Unable to route remaining labels via wall follower")
            candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
            _, _, _, label, _, stub_a, stub_b, moves = candidates[0]
            try:
                self._apply_path(stub_a, stub_b, label, moves)
            except RoutingConflict:
                invalid_paths[label].add(tuple(moves))
                continue
            invalid_paths.pop(label, None)
            self._remove_open_end(stub_a)
            self._remove_open_end(stub_b)
        self._post_routing_cleanup()

    def _post_routing_cleanup(self):
        for cell in list(self.cell_ports.keys()):
            self._update_tile_from_ports(cell)

    def _install_dual_tile(
        self,
        cell: Tuple[int, int],
        tile: int,
        stubs: List[Dict[str, object]],
        exit_map: Dict[str, str],
    ):
        r, c = cell
        self.tile_mat[r][c] = tile
        self.occupied_cells.add(cell)
        for stub in stubs:
            self._remove_open_end(stub)
        for stub in stubs:
            entry_side = stub["port"]  # type: ignore[index]
            label = stub["label"]  # type: ignore[index]
            exit_dir = exit_map[entry_side]
            nr = r + DIR_TO_DELTA[exit_dir][0]
            nc = c + DIR_TO_DELTA[exit_dir][1]
            self._ensure_in_bounds(nr, nc)
            new_stub = {
                "label": label,
                "cell": (nr, nc),
                "port": OPPOSITE[exit_dir],
            }
            self._register_open_end(new_stub)

    def _maybe_place_dual_tile(self, cell: Tuple[int, int]):
        stubs = self.open_ends_by_cell.get(cell)
        if not stubs or len(stubs) < 2:
            return
        if len(stubs) == 2:
            first, second = stubs
            if first["label"] == second["label"]:  # type: ignore[index]
                return
            d1 = first["port"]  # type: ignore[index]
            d2 = second["port"]  # type: ignore[index]
            for rule in DUAL_RULES:
                pair_a, pair_b = rule["pairs"]
                if (d1 in pair_a and d2 in pair_b) or (d1 in pair_b and d2 in pair_a):
                    self._install_dual_tile(cell, rule["tile"], [first, second], rule["exit"])
                    return
            return
        # Fallback to previous logic for four stubs
        by_label: Dict[int, List[str]] = defaultdict(list)
        for stub in stubs:
            by_label[stub["label"]].append(stub["port"])  # type: ignore[index]
        if len(by_label) != 2:
            return
        ports_per_label = list(by_label.items())
        if any(len(ports) != 2 for _, ports in ports_per_label):
            return
        shapes = frozenset(frozenset(ports) for _, ports in ports_per_label)
        tile = DUAL_ARC_TILES.get(shapes)
        if tile is None:
            return
        labels = [lbl for lbl, _ in ports_per_label]
        if self._labels_share_crossing(labels[0], labels[1]):
            return
        r, c = cell
        self.tile_mat[r][c] = tile
        self.occupied_cells.add(cell)
        self._clear_open_ends_at_cell(cell)

    def _place_crossing(
        self,
        cid: int,
        pos: Tuple[int, int],
        anchor_port: str,
        anchor_side: str,
    ) -> bool:
        r, c = pos
        self._ensure_in_bounds(r, c)
        existing_stubs = self.open_ends_by_cell.get(pos)
        if existing_stubs:
            crossing_labels = set(self.pd[cid])
            if any(stub["label"] not in crossing_labels for stub in existing_stubs):
                return False
        if pos in self.crossing_cells or pos in self.occupied_cells:
            return False
        mapping = port_mapping_from_anchor(anchor_port, anchor_side)
        self.orientation_map[cid] = mapping
        tile_id = decide_crossing_tile(anchor_port, anchor_side)
        self.tile_mat[r][c] = tile_id
        self.crossing_cells.add(pos)
        self.crossing_positions[cid] = pos
        self.placed_crossings.add(cid)
        self.occupied_cells.add(pos)
        return True

    def _register_crossing_stubs(
        self,
        cid: int,
        pos: Tuple[int, int],
        skip_ports: Optional[Set[str]] = None,
    ):
        mapping = self.orientation_map[cid]
        skip_ports = skip_ports or set()
        r, c = pos
        for port, label in zip(PORTS, self.pd[cid]):
            if port in skip_ports:
                continue
            direction = mapping[port]
            nr = r + DIR_TO_DELTA[direction][0]
            nc = c + DIR_TO_DELTA[direction][1]
            self._ensure_in_bounds(nr, nc)
            stub = {
                "label": label,
                "cell": (nr, nc),
                "port": OPPOSITE[direction],
            }
            self._register_open_end(stub)

    def _try_attach_crossing(self, remaining: Set[int], port_priority: Iterable[str]) -> bool:
        for port in port_priority:
            port_idx = PORT_INDEX[port]
            for label, stubs in list(self.open_ends_by_label.items()):
                if not stubs:
                    continue
                for stub in list(stubs):
                    cell = stub["cell"]  # type: ignore[index]
                    anchor_side = stub["port"]  # type: ignore[index]
                    if cell in self.occupied_cells:
                        continue
                    for cid, port_name in self.label_occ[label]:
                        if cid not in remaining or port_name != port:
                            continue
                        if not self._place_crossing(cid, cell, port, anchor_side):
                            continue
                        self._remove_open_end(stub)
                        self._clear_open_ends_at_cell(cell)
                        self._register_crossing_stubs(cid, cell, skip_ports={port})
                        remaining.remove(cid)
                        return True
        return False

    def _find_seed_pair(self) -> Optional[Tuple[int, int, int]]:
        for cid, quad in enumerate(self.pd):
            label = quad[3]
            for other_cid, port in self.label_occ[label]:
                if other_cid == cid:
                    continue
                if port == "a":
                    return cid, other_cid, label
        return None

    def _place_seed_pair(
        self,
        first: int,
        second: int,
        label: int,
        remaining: Set[int],
    ):
        if not self._place_crossing(first, self.center, "a", "N"):
            raise ValueError("Unable to place seed crossing at center")
        self._register_crossing_stubs(first, self.center)
        remaining.discard(first)
        stubs = self.open_ends_by_label.get(label)
        if not stubs:
            raise ValueError("Seed crossing failed to expose required open end")
        anchor_stub = stubs[0]
        cell = anchor_stub["cell"]  # type: ignore[index]
        anchor_side = anchor_stub["port"]  # type: ignore[index]
        if not self._place_crossing(second, cell, "a", anchor_side):
            raise ValueError("Unable to place second seed crossing adjacent to first")
        self._remove_open_end(anchor_stub)
        self._clear_open_ends_at_cell(cell)
        self._register_crossing_stubs(second, cell, skip_ports={"a"})
        remaining.discard(second)

    def _place_single_seed(self, first: int, remaining: Set[int]):
        if not self._place_crossing(first, self.center, "a", "N"):
            raise ValueError("Unable to place seed crossing at center")
        self._register_crossing_stubs(first, self.center)
        remaining.discard(first)

    def place_crossings(self):
        if not self.pd:
            return
        remaining: Set[int] = set(range(len(self.pd)))
        seed = self._find_seed_pair()
        if seed is not None:
            first, second, label = seed
            self._place_seed_pair(first, second, label, remaining)
        else:
            first = min(remaining)
            self._place_single_seed(first, remaining)

        while remaining:
            placed = self._try_attach_crossing(remaining, ("a", "d", "b", "c"))
            if placed:
                continue
            raise ValueError("Unable to place remaining crossings without disconnecting the diagram")

    def build(self) -> List[List[int]]:
        self.place_crossings()
        self.route_open_ends()
        trimmed = trim_tile_matrix(self.tile_mat)
        validate_tile_matrix(trimmed)
        padded = pad_tile_matrix(trimmed, pad=1)
        return fill_empty_tiles(padded)


def mosaic_from_pd(pd: List[Tuple[int, int, int, int]]):
    builder = MosaicBuilder(pd)
    return builder.build()


def mosaic_from_pd_auto(pd_str: str, attempts: int = 8) -> List[List[int]]:
    pd = parse_pd_string(pd_str)
    last_error: Optional[Exception] = None
    rng = random.Random(hash(pd_str))
    for attempt in range(attempts):
        candidate = list(pd)
        if attempt:
            rng.shuffle(candidate)
        try:
            return mosaic_from_pd(candidate)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError("Failed to generate mosaic: no attempts were executed")


def render_mosaic_image(
    tile_mat,
    tiles_dir,
    out_path=None,
    filler_tile=1,
    background=(255, 255, 255),
):
    if not tile_mat:
        raise ValueError("Tile matrix is empty; nothing to render")

    height = len(tile_mat)
    width = len(tile_mat[0])
    cache = {}

    def load_tile(tile_id: int):
        tid = tile_id if tile_id != 0 else filler_tile
        if tid in cache:
            return cache[tid]
        filename = f"tile_{tid:02d}.png"
        path = os.path.join(tiles_dir, filename)
        if not os.path.exists(path):
            if tid == filler_tile:
                raise FileNotFoundError(f"Missing filler tile: {path}")
            return load_tile(filler_tile)
        img = Image.open(path).convert("RGBA")
        cache[tid] = img
        return img

    sample = load_tile(filler_tile)
    tile_w, tile_h = sample.size
    canvas = Image.new("RGBA", (width * tile_w, height * tile_h), (0, 0, 0, 0))

    for r in range(height):
        for c in range(width):
            tile_id = tile_mat[r][c]
            img = load_tile(tile_id)
            x = c * tile_w
            y = r * tile_h
            canvas.paste(img, (x, y), img)

    if background is not None:
        bg = Image.new("RGB", canvas.size, background)
        bg.paste(canvas, mask=canvas.split()[3])
        result = bg
    else:
        result = canvas

    if out_path is not None:
        result.save(out_path)

    return result


if __name__ == "__main__":
    pd_str = "PD[X[14,2,15,1],X[24,18,1,17],X[2,8,3,7],X[6,4,7,3],X[4,16,5,15],X[16,6,17,5],X[8,23,9,24],X[20,9,21,10],X[10,19,11,20],X[18,11,19,12],X[12,21,13,22],X[22,13,23,14]]"
    matrix = mosaic_from_pd_auto(pd_str)
    print("Tile matrix (generated):")
    for row in matrix:
        print(row)

# 

# PD[X[1, 4, 2, 5], X[7, 10, 8, 11], X[3, 9, 4, 8], X[9, 3, 10, 2], X[5, 12, 6, 1], X[11, 6, 12, 7]]