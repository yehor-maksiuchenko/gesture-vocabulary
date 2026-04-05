import numpy as np

class GestureRecognizer:
    """
    Compares incoming hand data against saved gestures.
    Uses simple nearest-neighbor distance — no ML needed to start.
    """

    def __init__(self, db, threshold_static=0.8, threshold_dynamic=1.5):
        self.db = db
        self.threshold_static = threshold_static
        self.threshold_dynamic = threshold_dynamic

    def match_static(self, vector):
        """
        Compare a single frame's vector against all saved static gestures.
        Returns the best matching name, or None if no match is close enough.
        """
        candidates = self.db.get_all_static()
        if not candidates:
            return None

        best_name, best_dist = None, float('inf')
        for name, saved_vec in candidates:
            dist = np.linalg.norm(np.array(vector) - np.array(saved_vec))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        return best_name if best_dist < self.threshold_static else None

    def match_dynamic(self, sequence):
        """
        Compare a sequence of vectors against saved dynamic gestures.
        Uses DTW-lite: resamples both sequences to same length, then compares.
        Returns best matching name, or None.
        """
        candidates = self.db.get_all_dynamic()
        if not candidates:
            return None

        query = self._resample(sequence, 30)
        best_name, best_dist = None, float('inf')

        for name, saved_seq in candidates:
            reference = self._resample(saved_seq, 30)
            dist = np.mean([
                np.linalg.norm(np.array(q) - np.array(r))
                for q, r in zip(query, reference)
            ])
            if dist < best_dist:
                best_dist = dist
                best_name = name

        return best_name if best_dist < self.threshold_dynamic else None

    def _resample(self, sequence, target_len):
        """Resample a sequence to a fixed length by linear interpolation."""
        seq = np.array(sequence)
        indices = np.linspace(0, len(seq)-1, target_len)
        return [seq[int(i)].tolist() for i in indices]