from .tatqa import TatqaDataset
from .finqa import FinqaDataset
from .fetaqa import FetaqaDataset
from .fetaqa_perturbed import FetaqaPerturbedDataset
from .mmqa import MmqaDataset

DATASETS = {
    "tatqa": TatqaDataset,
    "finqa": FinqaDataset,
    "fetaqa": FetaqaDataset,
    "fetaqa_perturbed": FetaqaPerturbedDataset,
    "mmqa": MmqaDataset,
} 