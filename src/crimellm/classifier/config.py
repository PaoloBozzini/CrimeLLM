from dataclasses import dataclass, field


@dataclass
class Config:
    model_name: str = "law-ai/InLegalBERT"
    max_len: int = 256
    num_train_epochs: int = 4
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    output_dir: str = "./artifacts/checkpoints"
    seed: int = 42
    test_size: float = 0.33
    # If True, freeze encoder and train only the classification head (linear probe).
    # Faster, less data-hungry, lower ceiling. Set False for full fine-tune.
    freeze_encoder: bool = True
    id2label: dict[int, str] = field(default_factory=lambda: {0: "no", 1: "yes", 2: "unclear"})

    @property
    def label2id(self) -> dict[str, int]:
        return {v: k for k, v in self.id2label.items()}

    @property
    def num_labels(self) -> int:
        return len(self.id2label)
