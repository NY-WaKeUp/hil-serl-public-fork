import threading
from typing import Iterable, Optional

import numpy as np
from flax.core import frozen_dict

from serl_launcher.data.dataset import _sample
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore


class ShortEpisodeWeightedDemoBuffer(MemoryEfficientReplayBufferDataStore):
    """Prefer complete success episodes that are both shorter and newer."""

    def __init__(self, *args, recency_exponent: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.recency_exponent = recency_exponent
        self._episode_lengths = np.ones(self._capacity, dtype=np.int64)
        self._insertion_orders = np.zeros(self._capacity, dtype=np.int64)
        self._weights = None
        self._sample_lock = threading.Lock()

    def insert(self, data_dict, episode_length: int = 1, insertion_order: int = 0):
        with self._sample_lock:
            valid_before = self._is_correct_index.copy()
            super().insert(data_dict)
            new_valid = self._is_correct_index & ~valid_before
            self._episode_lengths[new_valid] = episode_length
            self._insertion_orders[new_valid] = insertion_order
            self._weights = None

    def _rebuild_weights(self) -> None:
        episode_lengths = self._episode_lengths[:len(self)]
        insertion_orders = self._insertion_orders[:len(self)]
        newest_order = insertion_orders.max(initial=0)
        freshness_ranks = newest_order - insertion_orders + 1
        # Per-transition mass 1 / (rank^alpha * L^2) gives an episode total
        # mass 1 / (rank^alpha * L): newer and shorter episodes are preferred.
        weights = 1.0 / (freshness_ranks**self.recency_exponent * episode_lengths**2)
        weights[~self._is_correct_index[:len(self)]] = 0.0
        total = weights.sum()
        self._weights = weights / total if total > 0 else np.full(len(self), 1 / len(self))

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> frozen_dict.FrozenDict:
        with self._sample_lock:
            if self._weights is None or len(self._weights) != len(self):
                self._rebuild_weights()
            indx = self.np_random.choice(len(self), size=batch_size, p=self._weights, replace=True)
            for index in range(batch_size):
                while not self._is_correct_index[indx[index]]:
                    indx[index] = self.np_random.choice(len(self), p=self._weights)
            if keys is None:
                keys = self.dataset_dict.keys()
            else:
                assert "observations" in keys

            keys = list(keys)
            keys.remove("observations")
            batch = MemoryEfficientReplayBufferDataStore.__mro__[2].sample(
                self, batch_size, keys, indx
            ).unfreeze()
            obs_keys = list(self.dataset_dict["observations"].keys())
            for pixel_key in self.pixel_keys:
                obs_keys.remove(pixel_key)
            batch["observations"] = {
                key: _sample(self.dataset_dict["observations"][key], indx)
                for key in obs_keys
            }
            for pixel_key in self.pixel_keys:
                obs_pixels = np.lib.stride_tricks.sliding_window_view(
                    self.dataset_dict["observations"][pixel_key], self._num_stack + 1, axis=0
                )[indx - self._num_stack].transpose((0, 4, 1, 2, 3))
                if pack_obs_and_next_obs:
                    batch["observations"][pixel_key] = obs_pixels
                else:
                    batch["observations"][pixel_key] = obs_pixels[:, :-1, ...]
                    if "next_observations" in keys:
                        batch["next_observations"][pixel_key] = obs_pixels[:, 1:, ...]
            return frozen_dict.freeze(batch)