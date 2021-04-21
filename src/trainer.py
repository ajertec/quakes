import os
import math
import logging
from packaging import version

from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import numpy as np
from tqdm import trange, tqdm

from transformers import AdamW

from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_scheduler
from transformers.trainer_utils import set_seed


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        evaluate_during_training: bool,
        train_batch_size: int,
        eval_batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        num_train_epochs: int,
        weight_decay: float,
        lr_scheduler_type: str,
        max_gradient_norm: float,
        logging_steps: int,
        evaluate_steps: int,
        output_dir: str,
        num_workers: int,
        fp16: bool,
        device: torch.device,
        seed: int,
    ):
        self.model = model
        self.model.to(device)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.evaluate_during_training = evaluate_during_training
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.max_gradient_norm = max_gradient_norm

        self.logging_steps = logging_steps
        self.evaluate_steps = evaluate_steps

        self.fp16 = fp16
        if self.fp16:
            raise NotImplementedError
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self.device = device

        self.seed = seed
        set_seed(self.seed)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.num_workers = num_workers

    def get_sampler(self, dataset, data_split):
        if data_split == "train":
            return RandomSampler(dataset)
        elif data_split == "eval":
            return SequentialSampler(dataset)
        else:
            raise ValueError(f"Sampler for data_split {data_split} not implemented.")

    def get_dataloader(self, dataset, data_split):

        sampler = self.get_sampler(dataset, data_split=data_split)

        if data_split == "train":
            batch_size = self.train_batch_size
        elif data_split == "eval":
            batch_size = self.eval_batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def create_optimizer(
        self,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):

        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters,
            betas=(beta1, beta2),
            eps=epsilon,
            lr=self.learning_rate,
        )

    def create_lr_scheduler(
        self,
        optimizer,
        num_training_steps,
        warmup_steps=0,
    ):
        return get_scheduler(
            self.lr_scheduler_type,
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def prepare_inputs(
        self,
        inputs,
    ) -> Dict[str, Union[torch.Tensor, Any]]:

        inputs_prepared = {}

        inputs_prepared["inputs"] = inputs[0].to(self.device)
        inputs_prepared["labels"] = inputs[1].to(self.device)

        return inputs_prepared

    def compute_loss(self, model, inputs):

        outputs = model(**inputs)
        loss = outputs[0]

        return loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:

        model.train()
        inputs = self.prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()

    def log(
        self,
        global_step,
        epoch,
        tr_loss,
    ):
        logs = {}

        logs["global_step"] = global_step
        logs["epoch"] = round(epoch, 2)
        logs["average_loss"] = round(tr_loss.item() / global_step, 4)

        last_lr = (
            # backward compatibility for pytorch schedulers
            self.lr_scheduler.get_last_lr()[0]
            if version.parse(torch.__version__) >= version.parse("1.4")
            else self.lr_scheduler.get_lr()[0]
        )
        logs["learning_rate"] = last_lr

        logging.info(logs)

    def prediction_loop(
        self,
        dataloader,
        model,
        has_labels: bool,
    ):
        num_examples = len(dataloader.dataset)
        # TODO de-hard-code
        all_preds = np.zeros(num_examples, 2)

        if has_labels:
            all_labels = np.zeros(num_examples, 2)

        model.eval()
        i = 0
        for inputs in dataloader:

            with torch.no_grad():
                inputs_prepared = self.prepare_inputs(inputs)
                outputs = model(**inputs_prepared)

            if has_labels:
                # loss is the first element in the outputs,
                # so we take second element-- logits
                logits = outputs[1]
            else:
                logits = outputs[0]

            batch_size = logits.shape[0]
            all_preds[i : batch_size + i] = logits.cpu().numpy().argmax(-1)

            if has_labels:
                all_labels[i : batch_size + i] = inputs_prepared["labels"].cpu().numpy()

            i += batch_size

        if has_labels:
            return all_preds, all_labels
        else:
            return all_preds

    def compute_metrics(
        self,
        preds,
        labels,
    ):
        metrics = {}

        acc = np.prod(preds == labels, -1) / preds.shape[0]
        acc_x = (preds[:, 0] == labels[:, 0]).sum() / preds.shape[0]
        acc_y = (preds[:, 1] == labels[:, 1]).sum() / preds.shape[0]

        metrics["acc"] = acc
        metrics["acc_x"] = acc_x
        metrics["acc_y"] = acc_y

        return metrics

    def evaluate(
        self,
        eval_dataloader,
    ):
        eval_preds, eval_labels = self.prediction_loop(
            dataloader=eval_dataloader, model=self.model, has_labels=True
        )

        metrics = self.compute_metrics(preds=eval_preds, labels=eval_labels)

        logging.info("Eval metrics: ")
        logging.info(metrics)

    def save_model(
        self,
    ):
        pass

    def train(
        self,
    ):

        # DATALOADERS:
        train_dataloader = self.get_dataloader(self.train_dataset, "train")

        if self.evaluate_during_training:
            eval_dataloader = self.get_dataloader(self.train_dataset, "eval")

        #
        num_update_steps_per_epoch = (
            len(train_dataloader) // self.gradient_accumulation_steps
        )
        max_steps = math.ceil(num_update_steps_per_epoch * self.num_train_epochs)
        total_train_batch_size = (
            self.train_batch_size * self.gradient_accumulation_steps
        )
        num_examples = len(train_dataloader.dataset)
        num_batches = len(train_dataloader)

        # TODO implement continue training...
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # OPTIMIZERS:
        optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler(
            optimizer, num_training_steps=max_steps
        )

        logging.info("*** Running training ***")
        logging.info(f"Number of examples: {num_examples}")
        logging.info(f"Number of batches: {num_batches}")
        logging.info(f"Number of update steps per epoch: {num_update_steps_per_epoch}")
        logging.info(f"Batch size per device: {self.train_batch_size}")
        logging.info(f"Total batch size: {total_train_batch_size}")
        logging.info(f"Number of training epochs: {self.num_train_epochs}")
        logging.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logging.info(f"Total steps during training: {max_steps}")

        tr_loss = torch.tensor(0.0).to(self.device)

        self.model.zero_grad()

        train_iterator = trange(
            epochs_trained,
            self.num_train_epochs,
            desc="Epoch",
        )

        for epoch in train_iterator:

            epoch_iterator = tqdm(
                train_dataloader,
                desc="Iteration",
            )

            for step, inputs in enumerate(epoch_iterator):

                tr_loss += self.training_step(self.model, inputs)

                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(train_dataloader) <= self.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
                ):
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_gradient_norm
                    )

                    if self.fp16:
                        raise NotImplementedError
                    else:
                        optimizer.step()

                    self.lr_scheduler.step()

                    self.model.zero_grad()

                    global_step += 1
                    log_epoch = epoch + (step + 1) / len(train_dataloader)

                    epoch_iterator.set_description(
                        "Avg loss: {:.9f}".format(tr_loss / global_step)
                    )

                    if global_step % self.logging_steps == 0:
                        self.log(global_step, log_epoch, tr_loss)

                    if (
                        global_step % self.evaluate_steps == 0
                        and self.evaluate_during_training
                    ):
                        self.evaluate(eval_dataloader)

    def freeze_parameters(
        self,
    ):
        pass

    def warmup_frozen_parameters(
        self,
    ):
        pass
