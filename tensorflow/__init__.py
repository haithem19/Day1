from types import SimpleNamespace

from . import keras


def _set_seed(seed):
	keras.utils.set_random_seed(seed)


random = SimpleNamespace(set_seed=_set_seed)
