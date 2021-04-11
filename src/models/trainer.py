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

from transformers import AdamW

from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_scheduler
from transformers.trainer_utils import set_seed


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        evaluate_during_training: bool,
        train_batch_size,
        eval_batch_size,
        gradient_accumulation_steps,
        learning_rate,
        num_train_epochs,
        weight_decay,
        lr_scheduler_type,
        max_gradient_norm,
        logging_steps,
        evaluate_steps,
        output_dir,
        num_workers,
        fp16,
        device,
        seed,
    ):
        self.model = model
        self.model.to(device)

        self.tokenizer = tokenizer

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
        if self.fp16 is not None:
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
        warmup_steps=None,
    ):
        return get_scheduler(
            self.lr_scheduler_type,
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def prepare_inputs(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Dict[str, Union[torch.Tensor, Any]]:

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        return inputs

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

        return logs

    def prediction_loop(
        self,
        dataloader,
        model,
        has_labels: bool,
    ):
        num_examples = len(dataloader.dataset)
        # TODO de-hard-code
        all_logits = np.zeros(num_examples, 2, model.x_size)

        model.eval()
        i = 0
        for inputs in dataloader:

            with torch.no_grad():
                inputs = self.prepare_inputs(inputs)
                outputs = model(**inputs)

            if has_labels:
                # loss is first element in the outputs
                logits = outputs[1]
            else:
                logits = outputs[0]

            batch_size = logits.shape[0]
            all_logits[i : batch_size + i] = logits.cpu().numpy()

            i += batch_size

        return all_logits

    def compute_metrics(
        self,
    ):
        pass

    def evaluate(
        self,
        eval_dataloader,
    ):
        pass

    def save_model(
        self,
    ):
        pass

    def train(
        self,
    ):

        # DATALOADERS:
        train_dataloader = self.get_dataloader(self.train_dataset, "train")

        if self.evaluate_during_training is not None:
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

        for epoch in range(epochs_trained, self.num_train_epochs):

            for step, inputs in enumerate(train_dataloader):

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

                    if global_step % self.logging_steps == 0:
                        self.log(global_step, log_epoch, tr_loss)

                    if global_step % self.evaluate_steps == 0:
                        self.evaluate()
                        self.model.train()

    def freeze_parameters(
        self,
    ):
        pass

    def warmup_frozen_parameters(
        self,
    ):
        pass
