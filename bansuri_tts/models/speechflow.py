from typing import Any, Dict, Tuple

import torch
import random
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher


# torch.manual_seed(42)
def generate_mask(x, p_cond=0.5, mask_span_length=128):
    """
    Generate a mask for the input tensor x.
    
    Parameters:
    - x (Tensor): Input tensor of shape (num_frames, num_features).
    - p_cond (float): Probability of applying the mask to a frame.
    - mask_span_length (int): Minimum span length of frames to mask.
    
    Returns:
    - mask (Tensor): Mask tensor of shape (num_frames, num_features), with 1s indicating masked positions and 0s indicating unmasked positions.
    """
    batch_size, num_frames, num_features = x.size()
    mask = torch.ones(batch_size, num_features, num_frames).to(device=x.device, dtype=torch.int16)

    j = 0
    for i in range(batch_size):
      while j < num_frames:
          if  random.random() > p_cond:
              start_pos = max(0, j - mask_span_length // 2)
              end_pos = min(num_frames, j + mask_span_length // 2)
              mask[i, :, start_pos:end_pos] = 0
              j = end_pos
          else:
              j += mask_span_length // 2

    return mask

def mask_from_lens(seq_lengths, device=None):
    """
    Generates a mask tensor based on provided sequence lengths.

    Args:
        seq_lengths: A LongTensor of shape (batch_size,) containing sequence lengths.
        device: (optional) The device (CPU or GPU) to store the mask on. Defaults to None (current device).

    Returns:
        A BoolTensor of shape (batch_size, max_seq_length) with True values for valid positions
        and False for padded positions in each sequence.
    """

    max_seq_length = seq_lengths.max()

    # Create a sequence of increasing indices (up to max_seq_length)
    range_of_elements = torch.arange(0, max_seq_length, dtype=seq_lengths.dtype, device=seq_lengths.device)

    # Expand the seq_lengths to broadcast along the row dimension
    mask = range_of_elements < seq_lengths.unsqueeze(1)

    return mask


class SpeechFlow(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `SpeechFlow`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        # optimizer = torch.optim.Adam(model.parameters())
        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

        # loss function
        self.criterion = torch.nn.MSELoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_loss_best = MinMetric()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, t, y, mask)
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x1, seq_lens = batch["audio_signals"].signal, batch["audio_lens"]
        y = x1.clone()
        mask = mask_from_lens(seq_lens, device=x1.device)
        x0 = torch.randn_like(x1).to(x1)
        t, xt, ut, _, y1 = self.FM.guided_sample_location_and_conditional_flow(x0, x1, y1=y)
        logits = self.forward(xt, t, y, mask)
        loss = self.criterion(logits, y)
        return loss, logits, y
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        # self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}