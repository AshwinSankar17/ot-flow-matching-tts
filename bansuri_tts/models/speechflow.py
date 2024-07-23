from typing import Any, Dict, Tuple, Optional

import torch
import numpy as np
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MinMetric, MeanMetric

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0
) -> torch.tensor:
    """ Adapted from WavLM
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return torch.from_numpy(mask)

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

        self.val_loss = []
        self.test_loss = []
    
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
        # self.val_loss.reset()
        # self.val_loss_best.reset()
        pass
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.net, norm_type=2)
        self.log_dict(norms)

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # prep data
        x1, seq_lens = batch["mel_spec"], batch["mel_spec_len"]
        y = x1.clone()
        # generate length mask and random content mask
        len_mask = mask_from_lens(seq_lens, device=x1.device).unsqueeze(1)
        x_mask = compute_mask_indices((x1.size(0), x1.size(2)), None, 0.85, 128, min_masks=2).to(x1.device).unsqueeze(1)
        # print(x1.size(), x_mask.size())
        x1 = x1 * x_mask
        # model step
        x0 = torch.randn_like(x1)
        t, xt, ut, _, y1 = self.FM.guided_sample_location_and_conditional_flow(x0, x1, y1=y)
        logits = self.forward(xt.transpose(1, 2), t, y.transpose(1, 2), len_mask).transpose(1, 2)
        # neg mask to compute loss for MLM
        neg_x_mask = (~x_mask) * len_mask
        logits = logits * neg_x_mask
        y = y * neg_x_mask
        # loss computation
        loss = self.criterion(logits, y)
        return loss, logits, y
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        # self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", loss, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.val_loss.append(loss)
        # update and log metrics
        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_loss = sum(self.val_loss) / len(self.val_loss)
        self.val_loss.clear()
        self.log("val/loss", val_loss, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss.append(loss)
        # self.test_acc(preds, targets)
        # self.log("test/loss", self.test_loss, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_test_epoch_end(self) -> None:
        test_loss = sum(self.test_loss) / len(self.test_loss)
        self.test_loss.clear()
        self.log("test/loss", test_loss, prog_bar=True)
    
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