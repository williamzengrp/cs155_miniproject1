"""Training-fitted median imputation and standardization."""
import math
import statistics

FEATURES=['time_signature','speechiness','danceability','duration_ms','energy','mode','instrumentalness','valence','key','tempo','loudness','acousticness','liveness']


def number(value):
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (ValueError,TypeError):return None


class Preprocessor:
    def fit(self,rows):
        self.medians=[statistics.median(v for r in rows if (v:=number(r[c])) is not None) for c in FEATURES]
        x=self.impute(rows)
        self.means=[statistics.mean(r[j] for r in x) for j in range(len(FEATURES))]
        self.scales=[statistics.pstdev(r[j] for r in x) or 1. for j in range(len(FEATURES))]
        return self
    def impute(self,rows):
        return [[self.medians[j] if (v:=number(r[c])) is None else v for j,c in enumerate(FEATURES)] for r in rows]
    def transform(self,rows):
        return [[(v-self.means[j])/self.scales[j] for j,v in enumerate(r)] for r in self.impute(rows)]

