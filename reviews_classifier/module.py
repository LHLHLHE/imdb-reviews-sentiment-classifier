import lightning as L
import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from transformers import AutoModel, get_linear_schedule_with_warmup


class BertSentimentModule(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.bert = AutoModel.from_pretrained(cfg.model.pretrained_model_name)
        self.head = nn.Linear(self.bert.config.hidden_size, cfg.model.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        for param in self.bert.parameters():
            param.requires_grad = False

        for layer in self.bert.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.train_acc = BinaryAccuracy()

        self.val_acc = BinaryAccuracy()
        self.val_precision_neg = BinaryPrecision()
        self.val_recall_neg = BinaryRecall()
        self.val_f1_neg = BinaryF1Score()

        self.test_acc = BinaryAccuracy()
        self.test_precision_neg = BinaryPrecision()
        self.test_recall_neg = BinaryRecall()
        self.test_f1_neg = BinaryF1Score()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)

    def _shared_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        probs_neg = torch.softmax(logits, dim=-1)[:, 1]
        labels = batch["labels"]
        return loss, probs_neg, labels

    def on_fit_start(self) -> None:
        self.train()

    def training_step(self, batch, batch_idx):
        loss, probs_neg, labels = self._shared_step(batch)
        accuracy = self.train_acc(probs_neg, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs_neg, labels = self._shared_step(batch)
        accuracy = self.val_acc(probs_neg, labels)
        precision = self.val_precision_neg(probs_neg, labels)
        recall = self.val_recall_neg(probs_neg, labels)
        f1 = self.val_f1_neg(probs_neg, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("val/precision_neg", precision, on_step=False, on_epoch=True)
        self.log("val/recall_neg", recall, on_step=False, on_epoch=True)
        self.log("val/f1_neg", f1, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        _, probs_neg, labels = self._shared_step(batch)
        accuracy = self.test_acc(probs_neg, labels)
        precision = self.test_precision_neg(probs_neg, labels)
        recall = self.test_recall_neg(probs_neg, labels)
        f1 = self.test_f1_neg(probs_neg, labels)

        self.log("test/accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("test/precision_neg", precision, on_step=False, on_epoch=True)
        self.log("test/recall_neg", recall, on_step=False, on_epoch=True)
        self.log("test/f1_neg", f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt_cfg = self.cfg.train.optimizer
        sch_cfg = self.cfg.train.scheduler

        bert_training_params = [p for p in self.bert.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": bert_training_params, "lr": float(opt_cfg.lr), "name": "bert_backbone"},
                {"params": self.head.parameters(), "lr": float(opt_cfg.lr) * 5.0, "name": "head"},
            ],
            weight_decay=float(opt_cfg.weight_decay),
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * float(sch_cfg.warmup_ratio))
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
