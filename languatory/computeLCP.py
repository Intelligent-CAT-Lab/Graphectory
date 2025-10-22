from __future__ import annotations
from typing import List, Tuple, Optional


def _contains_subsequence(seq: List[str], pat: Tuple[str, ...]) -> bool:
    """Check if pat is a subsequence of seq."""
    it = iter(seq)
    for tok in pat:
        for s in it:
            if s == tok:
                break
        else:
            return False
    return True


MIN_SUPPORT = 0.30
MAX_PERIOD_LEN = 12
LB_SUPPORT_DEFAULT = None


class PatternMiner:
    """
    Mine frequent sequential patterns on RLE sequences.

    For the top-1 longest pattern, compute per-position *support-consistent*
    streak lower bounds k_j:
      k_j = max k such that at least min_support of supporting sequences
            have a run length >= k at that position.

    Render as X_k. Minimal-period compression applies to the skeleton.

    Expected input format:
        List[dict] where each dict has:
        - "seq": List[str] - the sequence elements
        - "lens": List[int] - run lengths for each element
    """

    def __init__(
        self,
        min_support: float = MIN_SUPPORT,
        max_period_len: int = MAX_PERIOD_LEN
    ):
        self.min_support = float(min_support)
        self.max_period_len = int(max_period_len)

    @staticmethod
    def _mine_raw(sequences_rle: List[dict], min_support: float) -> List[Tuple[Tuple[str, ...], float]]:
        """Mine raw sequential patterns using GSP algorithm."""
        from gsppy.gsp import GSP

        if not sequences_rle:
            return []

        # Extract sequences
        seqs = [item["seq"] for item in sequences_rle if "seq" in item]
        if not seqs:
            return []
        if len(seqs) == 1:
            return [(tuple(seqs[0]), 1.0)]

        gsp = GSP(seqs)
        dicts = gsp.search(min_support=min_support)
        flat: List[Tuple[Tuple[str, ...], float]] = []
        for d in dicts:
            for pat, sup in d.items():
                flat.append((tuple(pat), float(sup)))
        return flat

    @staticmethod
    def _contains_subsequence_seq(seq: List[str], pat: Tuple[str, ...]) -> bool:
        """Check if pattern is a subsequence of seq."""
        return _contains_subsequence(seq, pat)

    @staticmethod
    def _first_match_indices(seq: List[str], pat: Tuple[str, ...]) -> Optional[List[int]]:
        """Return indices of the first subsequence match; None if absent."""
        idxs: List[int] = []
        start = 0
        for tok in pat:
            found = False
            for i in range(start, len(seq)):
                if seq[i] == tok:
                    idxs.append(i)
                    start = i + 1
                    found = True
                    break
            if not found:
                return None
        return idxs

    @staticmethod
    def _true_support_count(sequences_rle: List[dict], pat: Tuple[str, ...]) -> int:
        """Count how many sequences contain the pattern."""
        return sum(
            1 for item in sequences_rle
            if "seq" in item and PatternMiner._contains_subsequence_seq(item["seq"], pat)
        )

    @staticmethod
    def _percent(count: int, total: int) -> int:
        """Calculate percentage."""
        if total <= 0:
            return 0
        return int(round(100.0 * count / total))

    def _longest_by_len(self, mined: List[Tuple[Tuple[str, ...], float]]) -> List[Tuple[Tuple[str, ...], float]]:
        """Filter to only the longest patterns."""
        if not mined:
            return []
        max_len = max(len(p) for p, _ in mined)
        return [(p, sup) for p, sup in mined if len(p) == max_len]

    def _lb_runs_for_pattern(self, sequences_rle: List[dict], pat: Tuple[str, ...]) -> List[int]:
        """
        Compute per-position lower bounds k_j as the **minimum run length**
        among all supporting sequences' matched runs (first match per sequence).
        """
        buckets: List[List[int]] = [[] for _ in range(len(pat))]
        for item in sequences_rle:
            if "seq" not in item or "lens" not in item:
                continue
            idxs = self._first_match_indices(item["seq"], pat)
            if idxs is None:
                continue
            for j, idx in enumerate(idxs):
                buckets[j].append(int(item["lens"][idx]))

        lbs: List[int] = []
        for arr in buckets:
            # Use the minimum among the common subsequences; default to 1 if empty
            lbs.append(min(arr) if arr else 1)
        return lbs

    # --------- Minimal-period compression on skeleton with annotations ---------
    def _max_repeat_k(self, seq: List[str], i: int, L: int) -> int:
        """Find maximum repetitions of a period starting at position i."""
        n = len(seq)
        if i + L > n:
            return 1
        pat = seq[i:i+L]
        k = 1
        pos = i + L
        while pos + L <= n and seq[pos:pos+L] == pat:
            k += 1
            pos += L
        return k

    def _format_with_annots(self, seq: List[str], annots: List[str]) -> List[str]:
        """
        Compress repeated minimal periods on the sequence skeleton, but display
        each position as element + annotation (e.g., V_3). Annotations repeat with the period.
        """
        def tok(pos: int) -> str:
            return f"{seq[pos]}{annots[pos]}" if annots[pos] else seq[pos]

        out: List[str] = []
        n = len(seq)
        i = 0
        while i < n:
            best_L, best_k = 1, 1
            maxL = min(self.max_period_len, n - i)
            for L in range(1, maxL + 1):
                k = self._max_repeat_k(seq, i, L)
                if k > best_k or (k == best_k and L < best_L):
                    best_L, best_k = L, k

            if best_k >= 2:
                inner = ", ".join(tok(i + t) for t in range(best_L))
                out.append(f"({inner})^{best_k}")
                i += best_L * best_k
            else:
                out.append(tok(i))
                i += 1
        return out

    def format_pattern_with_lbs(self, pat: Tuple[str, ...], lbs: List[int]) -> str:
        """Format pattern with lower bound annotations."""
        if not pat:
            return "—"
        # '_k' only if k >= 2; keep tokens clean when the guarantee is 1
        annots = [f"_{k}" if k >= 2 else "" for k in lbs]
        comp = self._format_with_annots(list(pat), annots)
        return "⟨" + ", ".join(comp) + "⟩"

    # ---------- Public: top-1 pattern with % and LB exponents ----------
    def longest_ranked_top1(self, sequences_rle: List[dict]) -> Optional[Tuple[Tuple[str, ...], int, List[int]]]:
        """
        Find the longest, most frequent pattern.

        Returns:
            (pattern, percentage, lower_bounds) or None
        """
        if not sequences_rle:
            return None

        mined = self._mine_raw(sequences_rle, self.min_support)
        if not mined:
            return None

        cands = [p for p, _ in self._longest_by_len(mined)]
        totals = len(sequences_rle)

        scored: List[Tuple[Tuple[str, ...], int]] = []
        for p in cands:
            cnt = self._true_support_count(sequences_rle, p)
            pct = self._percent(cnt, totals)
            scored.append((p, pct))

        if not scored:
            return None

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        top_pat, top_pct = scored[0]

        lbs = self._lb_runs_for_pattern(sequences_rle, top_pat)
        return top_pat, top_pct, lbs
