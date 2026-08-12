"""BLOSUM62-derived substitution distribution for the hybrid corruption's substitution channel.

Borrows EvoDiff's `blosum62-special-MSA.mat`. That file is in BLOSUM column order
(A R N D C Q E G H I L K M F P S T W Y V B Z X J O U -), which is NOT our token order, so we
reorder into our canonical 20-AA alphabet (AA below, ids 0..19 = EvoDiff vocab.txt's first 20).

We only need the 20x20 standard-AA block. A BLOSUM score is a log-odds substitution score, so
`sub_probs[i,j] = softmax_j!=i(BLOSUM[i,j] / temp)` turns it into a row-stochastic "if residue i is
substituted, what does it become" distribution that favours biochemically similar residues. temp->inf
recovers the uniform default; small temp peaks toward the most-similar residue.
"""
from __future__ import annotations
import torch

# Canonical amino-acid alphabet: token id == index here (matches EvoDiff vocab.txt first 20).
AA = "ACDEFGHIKLMNPQRSTVWY"


def load_blosum62(path: str):
    """Parse a BLOSUM .mat file into {(row_aa, col_aa): score} plus the column-label list."""
    labels = None
    score = {}
    with open(path) as f:
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            parts = line.split()
            if labels is None:
                labels = parts                          # header row of column labels
                continue
            rlabel, vals = parts[0], [int(x) for x in parts[1:]]
            for clabel, v in zip(labels, vals):
                score[(rlabel, clabel)] = v
    return score, labels


def blosum_sub_probs(path: str, temp: float = 1.0, aa: str = AA) -> torch.Tensor:
    """(len(aa), len(aa)) row-stochastic substitution matrix, zero diagonal, BLOSUM-weighted."""
    score, _ = load_blosum62(path)
    n = len(aa)
    S = torch.full((n, n), -1e9)
    for i, ai in enumerate(aa):
        for j, aj in enumerate(aa):
            if i != j:
                S[i, j] = score[(ai, aj)] / temp
    return torch.softmax(S, dim=1)                      # diagonal -> ~0; rows sum to 1


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/bryan/Documents/models/EvoDiff-d3pm-blosum-38M/blosum62-special-MSA.mat"
    for temp in (1.0, 2.0, 100.0):
        P = blosum_sub_probs(path, temp=temp)
        rowsum = P.sum(1)
        diag = P.diagonal().abs().max().item()
        # Most-likely substitution for a few residues (sanity: should be biochemically similar).
        top = {AA[i]: AA[int(P[i].argmax())] for i in (AA.index("A"), AA.index("I"), AA.index("K"), AA.index("F"))}
        print(f"temp={temp:5.0f} | rowsum∈[{rowsum.min():.4f},{rowsum.max():.4f}] diag_max={diag:.1e} "
              f"| A->{top['A']} I->{top['I']} K->{top['K']} F->{top['F']}")
