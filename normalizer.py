"""
Based on code from Marcin Andrychowicz
"""
import numpy as np
import torch
import pytorch_util as ptu

class Normalizer(object):
    def __init__(
            self,
            size,
            eps=1e-8,
            default_clip_range=np.inf,
            mean=0,
            std=1,
    ):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.sum = np.zeros(self.size, np.float32)
        self.sumsq = np.zeros(self.size, np.float32)
        self.count = np.ones(1, np.float32)
        self.mean = mean + np.zeros(self.size, np.float32)
        self.std = std * np.ones(self.size, np.float32)
        self.synchronized = True

    def update(self, v):
        if v.ndim == 1:
            v = np.expand_dims(v, 0)
        assert v.ndim == 2
        assert v.shape[1] == self.size
        self.sum += v.sum(axis=0)
        self.sumsq += (np.square(v)).sum(axis=0)
        self.count[0] += v.shape[0]
        self.synchronized = False

    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return np.clip((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return mean + v * std

    def synchronize(self):
        self.mean[...] = self.sum / self.count[0]
        self.std[...] = np.sqrt(
            np.maximum(
                np.square(self.eps),
                self.sumsq / self.count[0] - np.square(self.mean)
            )
        )
        self.synchronized = True


class IdentityNormalizer(object):
    def __init__(self, *args, **kwargs):
        pass

    def update(self, v):
        pass

    def normalize(self, v, clip_range=None):
        return v

    def denormalize(self, v):
        return v


class FixedNormalizer(object):
    def __init__(
            self,
            size,
            default_clip_range=np.inf,
            mean=0,
            std=1,
            eps=1e-8,
    ):
        assert std > 0
        std = std + eps
        self.size = size
        self.default_clip_range = default_clip_range
        self.mean = mean + np.zeros(self.size, np.float32)
        self.std = std + np.zeros(self.size, np.float32)
        self.eps = eps

    def set_mean(self, mean):
        self.mean = mean + np.zeros(self.size, np.float32)

    def set_std(self, std):
        std = std + self.eps
        self.std = std + np.zeros(self.size, np.float32)

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return np.clip((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return mean + v * std

    def copy_stats(self, other):
        self.set_mean(other.mean)
        self.set_std(other.std)


class TorchFixedNormalizer(FixedNormalizer):
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def normalize_scale(self, v):
        """
        Only normalize the scale. Do not subtract the mean.
        """
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            std = std.unsqueeze(0)
        return v / std

    def denormalize(self, v):
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std

    def denormalize_scale(self, v):
        """
        Only denormalize the scale. Do not add the mean.
        """
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            std = std.unsqueeze(0)
        return v * std
