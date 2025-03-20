from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class BertGlobalPointerModelArguments:
    """
    Abstract class for model arguments.
    """
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for initialization."}
    )
    entity_num: int = field(
        default=None,
        metadata={"help": "num of entities"}
    )
    inner_dim: int = field(
        default=16,
        metadata={"help": "inner_dim of globalpointer layer."}
    )
    rope: bool = field(
        default=True,
        metadata={"help": "If use rope as postional encoding"}
    )


@dataclass
class BertGlobalPointerTrainingArguments(TrainingArguments):
    train_data: str = field(default=False, metadata={"help": "train data file path"})
