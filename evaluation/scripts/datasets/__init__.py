from .tatqa import TatqaDataset
from .finqa import FinqaDataset
from .fetaqa import FetaqaDataset
from .mmqa import MmqaDataset

DATASETS = {
    "tatqa": TatqaDataset,
    "finqa": FinqaDataset,
    "fetaqa": FetaqaDataset,
    "mmqa": MmqaDataset,
} 