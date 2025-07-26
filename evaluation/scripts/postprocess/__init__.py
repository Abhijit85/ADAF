from .tatqa import TatqaPostProcessor
from .fetaqa import FetaqaPostProcessor
from .finqa import FinqaPostProcessor

PROCESSORS = {
    "tatqa": TatqaPostProcessor,
    "fetaqa": FetaqaPostProcessor,
    "finqa": FinqaPostProcessor,
} 