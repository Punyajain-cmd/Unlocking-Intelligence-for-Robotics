"""
vision/scene_graph.py
──────────────────────
Builds and maintains a Scene Graph from a list of DetectedObject instances.

The graph captures:
  • Node per object (colour, shape, 3-D position)
  • Edges encoding spatial relationships (right_of, left_of, above, …)
  • Convenience query: find_by_relation(obj, relation) → object

The graph is rebuilt on every call to build() so it always reflects
the current scene state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from vision.object_detector import DetectedObject


# ──────────────────────────────────────────────────────────
# Spatial relation thresholds (metres)
# ──────────────────────────────────────────────────────────

DISTANCE_NEAR  = 0.10   # objects within this are "near"
AXIS_TOLERANCE = 0.04   # how collinear objects need to be for left/right/above/below


@dataclass
class SceneNode:
    obj: DetectedObject
    relations: Dict[str, List[int]] = field(default_factory=dict)
    # relations: {relation_name: [obj_id, ...]}


@dataclass
class SceneGraph:
    """
    Directed spatial graph over detected objects.

    Usage
    ─────
    >>> sg = SceneGraph()
    >>> sg.build(detected_objects)
    >>> ref  = sg.get_by_colour_shape("green", "cube")
    >>> subj = sg.find_by_relation(ref, "right_of")
    """

    nodes: Dict[int, SceneNode] = field(default_factory=dict)

    # ── Build ────────────────────────────────────────────────

    def build(self, objects: List[DetectedObject]) -> "SceneGraph":
        """Construct the graph from a fresh list of detections."""
        self.nodes = {o.id: SceneNode(obj=o) for o in objects}
        self._compute_relations(objects)
        self._tag_objects(objects)
        return self

    def _compute_relations(self, objects: List[DetectedObject]) -> None:
        """For every pair (a, b) compute directed spatial relations a→b."""
        for a in objects:
            node_a = self.nodes[a.id]
            for b in objects:
                if a.id == b.id:
                    continue
                rels = _spatial_relations(a.centre_3d, b.centre_3d)
                for rel in rels:
                    node_a.relations.setdefault(rel, []).append(b.id)

    def _tag_objects(self, objects: List[DetectedObject]) -> None:
        """Add spatial_tags (strings) to each DetectedObject."""
        positions = np.array([o.centre_3d for o in objects])  # (N, 3)
        if len(positions) == 0:
            return

        # Table-relative tags: leftmost / rightmost / front / back
        for o in objects:
            x, y, z = o.centre_3d
            # x axis: positive = right (in our sim frame)
            all_x = positions[:, 0]
            if x <= np.percentile(all_x, 25):
                o.spatial_tags.append("leftmost")
            if x >= np.percentile(all_x, 75):
                o.spatial_tags.append("rightmost")
            # y axis: negative = front
            all_y = positions[:, 1]
            if y <= np.percentile(all_y, 25):
                o.spatial_tags.append("front")
            if y >= np.percentile(all_y, 75):
                o.spatial_tags.append("back")
            # edge: near table boundary
            dist_from_centre = np.sqrt(x**2 + y**2)
            if dist_from_centre > 0.22:
                o.spatial_tags.append("edge")

    # ── Query ────────────────────────────────────────────────

    def get_by_id(self, obj_id: int) -> Optional[DetectedObject]:
        node = self.nodes.get(obj_id)
        return node.obj if node else None

    def get_by_colour_shape(
        self,
        colour: Optional[str],
        shape:  Optional[str],
    ) -> Optional[DetectedObject]:
        for node in self.nodes.values():
            if node.obj.matches(colour, shape):
                return node.obj
        return None

    def get_all(self) -> List[DetectedObject]:
        return [n.obj for n in self.nodes.values()]

    def find_by_relation(
        self,
        reference: DetectedObject,
        relation:  str,
    ) -> Optional[DetectedObject]:
        """
        Return the object that stands in `relation` to `reference`.

        Example: find_by_relation(green_cube, "right_of")
                 returns the object that is to the right of the green cube.
        """
        node = self.nodes.get(reference.id)
        if node is None:
            return None
        neighbours = node.relations.get(relation, [])
        if not neighbours:
            return None
        # Return the closest neighbour along the relevant axis
        ref_pos = np.array(reference.centre_3d)
        best, best_dist = None, float("inf")
        for nid in neighbours:
            nb = self.get_by_id(nid)
            if nb:
                d = float(np.linalg.norm(np.array(nb.centre_3d) - ref_pos))
                if d < best_dist:
                    best_dist, best = d, nb
        return best

    def compute_target_position(
        self,
        reference: DetectedObject,
        relation:  str,
        offset_m:  float = 0.08,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Compute a goal 3-D position that places an object
        in `relation` to `reference`.

        E.g. "to the right of the green cube" → position 8 cm to the right.
        """
        rx, ry, rz = reference.centre_3d
        offsets = {
            "right_of":    ( offset_m,  0.0,   0.0),
            "left_of":     (-offset_m,  0.0,   0.0),
            "above":       ( 0.0,        0.0,   offset_m),
            "on_top_of":   ( 0.0,        0.0,   offset_m * 1.2),
            "below":       ( 0.0,        0.0,  -offset_m),
            "in_front_of": ( 0.0,       -offset_m, 0.0),
            "behind":      ( 0.0,        offset_m, 0.0),
            "beside":      ( offset_m,   0.0,   0.0),
            "near":        ( offset_m * 0.5, offset_m * 0.5, 0.0),
        }
        dx, dy, dz = offsets.get(relation, (offset_m, 0.0, 0.0))
        return (rx + dx, ry + dy, rz + dz)

    def summary(self) -> str:
        lines = [f"SceneGraph ({len(self.nodes)} objects)"]
        for node in self.nodes.values():
            o = node.obj
            rel_strs = []
            for rel, ids in node.relations.items():
                for oid in ids[:2]:   # truncate for readability
                    nb = self.get_by_id(oid)
                    if nb:
                        rel_strs.append(f"{rel}({nb.colour}_{nb.shape})")
            lines.append(
                f"  [{o.id}] {o.colour} {o.shape}  pos={o.centre_3d}  "
                f"tags={o.spatial_tags}  rels={rel_strs[:3]}"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# Spatial geometry helpers
# ──────────────────────────────────────────────────────────

def _spatial_relations(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> List[str]:
    """
    Given positions a and b (xyz in metres),
    return the list of relations that describe a relative to b.

    Convention (robot-centric, tabletop):
      x positive → right
      y positive → back  (away from robot)
      z positive → up
    """
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = ax - bx, ay - by, az - bz
    dist = float(np.sqrt(dx**2 + dy**2 + dz**2))

    rels: List[str] = []

    if dist < DISTANCE_NEAR:
        rels.append("near")

    # Horizontal plane (x axis = left/right)
    if abs(dx) > AXIS_TOLERANCE and abs(dx) > abs(dy):
        if dx > 0:
            rels.append("right_of")
        else:
            rels.append("left_of")

    # Horizontal plane (y axis = front/back)
    if abs(dy) > AXIS_TOLERANCE and abs(dy) > abs(dx):
        if dy > 0:
            rels.append("behind")
        else:
            rels.append("in_front_of")

    # Vertical (z axis)
    if abs(dz) > AXIS_TOLERANCE:
        if dz > 0:
            rels.append("above")
        else:
            rels.append("below")

    # Stack / on top of  (very close horizontally, clearly above)
    horiz_dist = float(np.sqrt(dx**2 + dy**2))
    if dz > AXIS_TOLERANCE and horiz_dist < DISTANCE_NEAR:
        rels.append("on_top_of")

    # Beside: near + roughly same height
    if dist < DISTANCE_NEAR * 1.5 and abs(dz) < AXIS_TOLERANCE:
        rels.append("beside")

    return rels


# ── self-test ──────────────────────────────────────────────

if __name__ == "__main__":
    from vision.object_detector import DetectedObject

    objects = [
        DetectedObject(0, "blue",   "block",  centre_3d=(-0.10, 0.00, 0.65)),
        DetectedObject(1, "green",  "cube",   centre_3d=( 0.00, 0.00, 0.65)),
        DetectedObject(2, "red",    "sphere", centre_3d=( 0.10, 0.00, 0.65)),
        DetectedObject(3, "yellow", "block",  centre_3d=( 0.00, 0.10, 0.65)),
    ]

    sg = SceneGraph()
    sg.build(objects)
    print(sg.summary())

    green = sg.get_by_colour_shape("green", "cube")
    print("\nObject to the right of green cube:",
          sg.find_by_relation(green, "right_of"))
    print("Object to the left of green cube:",
          sg.find_by_relation(green, "left_of"))

    goal = sg.compute_target_position(green, "right_of")
    print(f"\nGoal position (right_of green cube): {goal}")
