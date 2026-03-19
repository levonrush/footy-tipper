import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit


def _resolve_round_id_column(X: pd.DataFrame) -> str:
    if "round_id" in X.columns:
        return "round_id"

    # After ColumnTransformer with passthrough, round_id is commonly named this.
    if "remainder__round_id" in X.columns:
        return "remainder__round_id"

    candidate_cols = [col for col in X.columns if str(col).endswith("__round_id")]
    if len(candidate_cols) == 1:
        return candidate_cols[0]
    if len(candidate_cols) > 1:
        raise ValueError(
            "InSeasonSplit found multiple round_id-like columns: "
            + ", ".join(map(str, candidate_cols))
        )

    raise ValueError("InSeasonSplit requires X to include a 'round_id' column.")


class InSeasonSplit(BaseCrossValidator):
    """
    Cross-validator for in-season forecasting: within each season,
    split by round blocks to avoid train/test leakage within rounds.
    """

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def _season_rounds(self, X, groups):
        if groups is None:
            raise ValueError("InSeasonSplit requires 'groups' (competition_year) to be passed.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("InSeasonSplit requires X to include a 'round_id' column.")
        if len(groups) != len(X):
            raise ValueError("InSeasonSplit requires groups length to match X length.")

        round_col = _resolve_round_id_column(X)
        round_ids = pd.to_numeric(X[round_col], errors="coerce").to_numpy()
        seasons = np.unique(groups)
        season_rounds = []
        for season in seasons:
            idx = np.where(groups == season)[0]
            rounds = np.sort(np.unique(round_ids[idx][~np.isnan(round_ids[idx])]))
            if len(rounds) < 2:
                continue
            season_rounds.append((idx, rounds))
        return season_rounds

    def split(self, X, y=None, groups=None):
        round_col = _resolve_round_id_column(X)
        round_ids = pd.to_numeric(X[round_col], errors="coerce").to_numpy()
        for idx, season_rounds in self._season_rounds(X, groups):
            n_splits = min(self.n_splits, len(season_rounds) - 1)
            if n_splits < 2:
                continue

            tscv = TimeSeriesSplit(n_splits=n_splits)
            for train_rel, test_rel in tscv.split(season_rounds):
                train_rounds = season_rounds[train_rel]
                test_rounds = season_rounds[test_rel]

                train_idx = idx[np.isin(round_ids[idx], train_rounds)]
                test_idx = idx[np.isin(round_ids[idx], test_rounds)]
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None or groups is None:
            return 0

        total = 0
        for _, season_rounds in self._season_rounds(X, groups):
            n = min(self.n_splits, len(season_rounds) - 1)
            if n >= 2:
                total += n
        return total
