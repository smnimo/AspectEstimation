"""
Item attributes of the catalog (knowledge.csv)

Two formats:

- typed (two header rows): attribute names, then attribute kinds ('ordinal' or 'categorical'), then one row per item.
  An ordinal value is a level 1..N; a categorical cell holds zero or more values separated by ';' (integers 1..N or labels).
- legacy (no header, the format of the paper): integers 1..N, one value per item and attribute. Every attribute is read
  as a single-valued categorical attribute, which reproduces the computation of the MATLAB version.

Features: the values of all attributes, attribute-major (value v of attribute a -> offsets[a] + v), as the 15 features
(5 attributes x 3 values) of the paper.
"""

import csv
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

ORDINAL = 'ordinal'
CATEGORICAL = 'categorical'
KINDS = (ORDINAL, CATEGORICAL)


@dataclass
class Attribute:
    name: str
    kind: str                  # ORDINAL or CATEGORICAL
    labels: List[str]          # label of each value (index 0..n_values-1)

    @property
    def n_values(self) -> int:
        return len(self.labels)


@dataclass
class Knowledge:
    attributes: List[Attribute]
    values: List[List[Tuple[int, ...]]]  # values[item][attribute]: sorted 0-based value indices
    legacy: bool = False                 # read from a headerless (paper) file
    _offsets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._offsets = np.concatenate([[0], np.cumsum([a.n_values for a in self.attributes])]).astype(int)

    @property
    def n_items(self) -> int:
        return len(self.values)

    @property
    def n_attributes(self) -> int:
        return len(self.attributes)

    @property
    def n_values(self) -> np.ndarray:
        return np.array([a.n_values for a in self.attributes])

    @property
    def offsets(self) -> np.ndarray:
        """Index of the first feature of each attribute (length n_attributes + 1)"""
        return self._offsets

    @property
    def n_features(self) -> int:
        return int(self._offsets[-1])

    @property
    def kinds(self) -> List[str]:
        return [a.kind for a in self.attributes]

    def feature_labels(self) -> List[str]:
        return [f'{a.name} {label}' for a in self.attributes for label in a.labels]

    def item_values(self, items: Sequence[int]) -> List[List[Tuple[int, ...]]]:
        """Value sets of the given items (1-based ids, e.g. a .list of a session)"""
        return [self.values[int(i) - 1] for i in items]

    def set_features(self, value_sets: Sequence[Sequence[Tuple[int, ...]]]) -> np.ndarray:
        """(len(value_sets) x n_features) 0/1 matrix; value_sets[row][attribute] are 0-based value indices"""
        out = np.zeros((len(value_sets), self.n_features))
        for row, vals in enumerate(value_sets):
            for a, vs in enumerate(vals):
                out[row, self._offsets[a] + np.array(vs, dtype=int)] = 1
        return out

    def features(self, items: Sequence[int]) -> np.ndarray:
        """(len(items) x n_features) 0/1 matrix of the values of the given items (1-based ids)"""
        return self.set_features(self.item_values(items))

    def value_matrix(self) -> np.ndarray:
        """(n_items x n_attributes) values 1..N (only when every item has exactly one value per attribute)"""
        if any(len(vs) != 1 for item in self.values for vs in item):
            raise ValueError("a value matrix needs exactly one value per item and attribute")
        return np.array([[vs[0] + 1 for vs in item] for item in self.values], dtype=int)


def from_value_matrix(kb: np.ndarray, names: Optional[Sequence[str]] = None,
                      n_values: Optional[Sequence[int]] = None) -> Knowledge:
    """
    Legacy knowledge base: (items x attributes) integers 1..N as single-valued categorical attributes

    Args:
        n_values: number of values of each attribute (default: the maximum in kb)
    """
    kb = np.asarray(kb, dtype=int)
    if kb.min() < 1:
        raise ValueError("values must be integers >= 1")
    n_values = kb.max(axis=0) if n_values is None else np.asarray(n_values)
    names = names or [f'attribute {a + 1}' for a in range(kb.shape[1])]
    attributes = [Attribute(name, CATEGORICAL, [str(v) for v in range(1, int(n) + 1)])
                  for name, n in zip(names, n_values)]
    values = [[(int(v) - 1,) for v in row] for row in kb]
    return Knowledge(attributes, values, legacy=True)


def _is_header(row: Sequence[str]) -> bool:
    return any(not cell.strip().lstrip('-').isdigit() for cell in row if cell.strip())


def load_knowledge(path: str) -> Knowledge:
    """
    Read knowledge.csv (typed with two header rows, or legacy without header; see the module docstring)

    Raises:
        ValueError: unknown attribute kind, ordinal cell without exactly one level, or rows of different lengths
    """
    with open(path, newline='') as fp:
        rows = [row for row in csv.reader(fp) if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"{path} is empty")
    if not _is_header(rows[0]):
        return from_value_matrix(np.array([[int(c) for c in row] for row in rows]))

    names = [c.strip() for c in rows[0]]
    kinds = [c.strip().lower() for c in rows[1]]
    if len(kinds) != len(names) or any(k not in KINDS for k in kinds):
        raise ValueError(f"the second row of {path} must give one kind ({' / '.join(KINDS)}) per attribute")
    body = rows[2:]
    if any(len(row) != len(names) for row in body):
        raise ValueError(f"every item row of {path} needs {len(names)} cells")

    attributes, columns = [], []
    for a, (name, kind) in enumerate(zip(names, kinds)):
        cells = [[t.strip() for t in row[a].split(';') if t.strip()] for row in body]
        if kind == ORDINAL:
            if any(len(c) != 1 or not c[0].isdigit() or int(c[0]) < 1 for c in cells):
                raise ValueError(f"ordinal attribute '{name}' needs one level 1..N per item")
            n = max(int(c[0]) for c in cells)
            labels = [str(v) for v in range(1, n + 1)]
            columns.append([(int(c[0]) - 1,) for c in cells])
        else:
            tokens = [t for c in cells for t in c]
            if tokens and all(t.isdigit() and int(t) >= 1 for t in tokens):
                labels = [str(v) for v in range(1, max(int(t) for t in tokens) + 1)]
                index = {label: i for i, label in enumerate(labels)}
            else:
                labels = list(dict.fromkeys(tokens))  # order of first appearance
                index = {label: i for i, label in enumerate(labels)}
            columns.append([tuple(sorted({index[t] for t in c})) for c in cells])
        attributes.append(Attribute(name, kind, labels))
    values = [[columns[a][i] for a in range(len(names))] for i in range(len(body))]
    return Knowledge(attributes, values)
