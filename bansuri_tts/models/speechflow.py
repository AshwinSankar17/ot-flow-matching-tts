from typing import Any, Dict, Tuple, Optional

import torch
import random
import librosa
import numpy as np
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch import loggers as ptl_loggers
from torchmetrics import MinMetric, MeanMetric

import torchdiffeq
import matplotlib.pyplot as plt


def plot_spectrogram_to_numpy(spectrogram):
    spectrogram = spectrogram.astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(librosa.power_to_db(spectrogram), aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


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

def random_interval_masking(batch_size, length, *, min_size, min_count, max_count, device):
    tensor = torch.full((batch_size, length), False, device=device, dtype=torch.bool)
    for i in range(batch_size):

        # Expected sum of all intervals
        expected_length = random.randint(min_count, max_count)

        # Number of intervals
        num_intervals = random.randint(1, expected_length // min_size)

        # Generate interval lengths
        lengths = [min_size] * num_intervals
        for _ in range(expected_length - num_intervals * min_size):
            lengths[random.randint(0, num_intervals - 1)] += 1

        # Generate start points
        placements = []
        offset = 0
        remaining = expected_length
        for l in lengths:
            start_position = random.uniform(offset, remaining - l)
            placements.append(start_position)
            offset = start_position + l
            remaining -= l

        # Write to tensor
        for l, p in zip(lengths, placements):
            tensor[i, int(p):int(p + l)] = True

    return tensor

def generate_conditional_mask(bsz, p_keep, device="cpu"):
    return torch.rand(bsz, device=device) < p_keep

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
        self.sigma_min = 1e-5
        # optimizer = torch.optim.Adam(model.parameters())
        # self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        # self.FM = ConditionalFlowMatcher(sigma=0.0)

        # loss function
        # self.criterion = torch.nn.HuberLoss(reduction="none")
        self.criterion = torch.nn.MSELoss(reduction="none")

        self.val_loss = []
        self.test_loss = []
    
    @torch.no_grad()
    def forward(self, y: torch.Tensor, n_timesteps: int = 100, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        y = y.transpose(1, 2)
        traj = torchdiffeq.odeint(
            lambda t, x: self.net.forward(x, t.unsqueeze(0), y, mask),
            # torch.randn((1, 841, 128)),
            torch.randn_like(y),
            torch.linspace(0, 1, n_timesteps, device=y.device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        return traj
    
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

        # generate length mask and random content mask and conditional mask
        len_mask = mask_from_lens(seq_lens, device=x1.device).unsqueeze(1)
        x_mask = random_interval_masking(x1.size(0), x1.size(2), min_size=10, min_count=int(0.7 * x1.size(2)), max_count=x1.size(2), device=x1.device).unsqueeze(1)
        conditional_mask = generate_conditional_mask(x1.size(0), 0.8, x1.device).unsqueeze(1).unsqueeze(2)

        joint_mask = x_mask * conditional_mask

        y = y * joint_mask
        
        # Sample noise
        x0 = torch.randn_like(x1)
        # Sample timestep
        t = torch.rand([x1.size(0), 1, 1]).to(x1)
        # Compute Flow
        ut = x1 - (1 - self.sigma_min) * x0
        # Sample location w.r.t t
        xt = (1 - (1 - self.sigma_min) * t) * x0 + x1 * t

        # Predict flow
        vt = self.net(xt.transpose(1, 2), t.squeeze(), y.transpose(1, 2), len_mask).transpose(1, 2)
        
        # neg mask to compute loss for MLM
        neg_joint_mask = (~joint_mask) * len_mask
        # vt_masked = vt * neg_joint_mask
        # ut_masked = ut * neg_joint_mask
        neg_joint_mask = neg_joint_mask.repeat(1, x1.size(1), 1)
        loss = self.criterion(vt, ut)
        loss = loss[neg_joint_mask].mean()
        # loss = self.criterion(vt, ut).mean(dim=1)
        
        # loss = loss * neg_joint_mask
        # n_masked = neg_joint_mask.sum(dim=-1).clamp(min=1).squeeze(1)
        # loss = loss.sum(dim=-1) / n_masked
        # loss = loss.mean()

        return loss, vt, y
    
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

        # if batch_idx == 0:
        #     with torch.no_grad():
        #         x1 = batch["mel_spec"]
        #         len_mask = mask_from_lens(batch["mel_spec_len"], device=x1.device).unsqueeze(1)
        #         x_mask = compute_mask_indices((x1.size(0), x1.size(2)), None, 0.85, 128, min_masks=2).to(x1.device).unsqueeze(1)
        #         x1 = x1 * x_mask
        #         traj = self.forward(x1, 100, len_mask)

        #         y = traj[-1].transpose(1, 2)
        #         y = y * len_mask
        #         idx = torch.argmax(batch["mel_spec_len"])
        #         gt_image = batch["mel_spec"][idx]
        #         masked_image = x1[idx]
        #         gen_image = y[idx]
        #         self.log_wandb_image(
        #             key="train/gt_mel",
        #             images=[gt_image]
        #         )
        #         self.log_wandb_image(
        #             key="train/masked_mel",
        #             images=[masked_image]
        #         )
        #         self.log_wandb_image(
        #             key="train/gen_mel",
        #             images=[gen_image]
        #         )

        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x1 = batch["mel_spec"]
        len_mask = mask_from_lens(batch["mel_spec_len"], device=x1.device).unsqueeze(1)

        x_mask = random_interval_masking(x1.size(0), x1.size(2), min_size=10, min_count=int(0.7 * x1.size(2)), max_count=x1.size(2), device=x1.device).unsqueeze(1)
        conditional_mask = generate_conditional_mask(x1.size(0), 0.8, x1.device).unsqueeze(1).unsqueeze(2)
        joint_mask = x_mask * conditional_mask

        y = x1 * joint_mask
        traj = self.forward(y, 100, len_mask)

        y_pred = traj[-1].transpose(1, 2)
        neg_joint_mask = (~joint_mask) * len_mask
        neg_joint_mask = neg_joint_mask.repeat(1, x1.size(1), 1)
        # y_pred = y_pred * neg_joint_mask
        # x_true = x1 * neg_joint_mask

        # loss = self.criterion(x_true, y_pred) / (torch.sum(neg_joint_mask))
        loss = self.criterion(x1, y_pred)
        loss = loss[neg_joint_mask].mean()
        # loss = self.criterion(x1, y_pred).mean(dim=1)

        # loss = loss * neg_joint_mask
        # n_masked = neg_joint_mask.sum(dim=-1).clamp(min=1).squeeze(1)
        # loss = loss.sum(dim=-1) / n_masked
        # loss = loss.mean()

        idx = torch.argmax(batch["mel_spec_len"])
        if batch_idx == 0:
            gt_image = batch["mel_spec"][idx]
            masked_image = y[idx]
            gen_image = y_pred[idx]
            self.log_wandb_image(
                key="val/gt_mel",
                images=[gt_image]
            )
            self.log_wandb_image(
                key="val/masked_mel",
                images=[masked_image]
            )
            self.log_wandb_image(
                key="val/gen_mel",
                images=[gen_image]
            )
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
        self.log("val/loss", val_loss, sync_dist=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x1 = batch["mel_spec"]
        len_mask = mask_from_lens(batch["mel_spec_len"], device=x1.device).unsqueeze(1)

        x_mask = random_interval_masking(x1.size(0), x1.size(2), min_size=10, min_count=int(0.7 * x1.size(2)), max_count=x1.size(2), device=x1.device).unsqueeze(1)
        conditional_mask = generate_conditional_mask(x1.size(0), 0.8, x1.device).unsqueeze(1).unsqueeze(2)
        joint_mask = x_mask * conditional_mask

        y = x1 * joint_mask
        traj = self.forward(y, 100, len_mask)

        y_pred = traj[-1].transpose(1, 2)
        neg_joint_mask = (~joint_mask) * len_mask

        # x_true = x1 * neg_joint_mask
        # y_pred = y_pred * neg_joint_mask
        neg_joint_mask = neg_joint_mask.repeat(1, x1.size(1), 1)
        loss = self.criterion(x1, y_pred)
        loss = loss[neg_joint_mask].mean()
        # loss = self.criterion(x1, y_pred).mean(dim=1)

        # loss = loss * neg_joint_mask
        # n_masked = neg_joint_mask.sum(dim=-1).clamp(min=1).squeeze(1)
        # loss = loss.sum(dim=-1) / n_masked
        # loss = loss.mean()
        # loss = self.criterion(x_true, y_pred) / (torch.sum(neg_joint_mask))
        idx = torch.argmax(batch["mel_spec_len"])
        if batch_idx == 0:
            gt_image = batch["mel_spec"][idx]
            masked_image = y[idx]
            gen_image = y_pred[idx]
            self.log_wandb_image(
                key="val/gt_mel",
                images=[gt_image]
            )
            self.log_wandb_image(
                key="val/masked_mel",
                images=[masked_image]
            )
            self.log_wandb_image(
                key="val/gen_mel",
                images=[gen_image]
            )
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
        # stepping_batches = self.trainer.estimated_stepping_batches
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
    
    def log_wandb_image(self, images, key):
        wandb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, ptl_loggers.WandbLogger):
                wandb_logger = logger

        if wandb_logger is not None:
            images = [plot_spectrogram_to_numpy(image.detach().cpu().float().numpy()) for image in images]
            wandb_logger.log_image(key=key, images=images)