from .tatqa import TatqaPostProcessor
from .fetaqa import FetaqaPostProcessor
from .fetaqa_perturbed import FetaqaPerturbedPostProcessor
from .finqa import FinqaPostProcessor

PROCESSORS = {
    "tatqa": TatqaPostProcessor,
    "fetaqa": FetaqaPostProcessor,
    "fetaqa_perturbed": FetaqaPerturbedPostProcessor,
    "finqa": FinqaPostProcessor,
} 