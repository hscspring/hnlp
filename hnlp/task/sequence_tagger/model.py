from dataclasses import dataclass
from transfomers import BertForSequenceClassification
from transfomers import TFBertForSequenceClassification


@dataclass
class SequenceClassification:

    frame: str = "py"

    def __post_init__(self):
        if self.frame == "py":
            model = BertForSequenceClassification()