from dataclasses import dataclass
from typing import TypeVar, Generic

from transformers import BertModel

import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)




class BertPretrained(BertModel):

    pass