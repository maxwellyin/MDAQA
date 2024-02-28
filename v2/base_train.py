# %%
import torch
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import default_data_collator, RobertaConfig
from datasets import load_from_disk
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW, Adam
from accelerate import Accelerator
from util import preprocess_training_examples, preprocess_validation_examples, compute_metrics, SOURCE_DOMAIN, MODEL_CHECKPOINT, BATCH_SIZE
import roberta
# %%
DATA = f"../data/{SOURCE_DOMAIN}_train"
# %%
raw_datasets = load_from_disk(DATA)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
train_dataset = raw_datasets.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets.column_names,
)
train_dataset.set_format("torch")
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=BATCH_SIZE,
)
# %%
# config = RobertaConfig.from_pretrained("roberta-base")
# model = roberta.RobertaForQuestionAnswering(config)
model = roberta.RobertaBMW.from_pretrained("roberta-base")
# %%
params_base = [v for k, v in model.named_parameters() if "bottleMask" not in k]
params_m = [v for k, v in model.named_parameters() if "bottleMask" in k]
optimizer_base = AdamW(params_base, lr=2e-5)
optimizer_m = Adam(params_m, lr=1e-3)

accelerator = Accelerator(fp16=torch.cuda.is_available())
model, optimizer_base, optimizer_m, train_dataloader = accelerator.prepare(
    model, optimizer_base, optimizer_m, train_dataloader
)

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler_base = get_scheduler(
    "linear",
    optimizer=optimizer_base,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# %%
# Training
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_train_epochs):       
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer_base.step()
        lr_scheduler_base.step()
        optimizer_base.zero_grad()

        optimizer_m.step()
        optimizer_m.zero_grad()

        progress_bar.update(1)

# %%
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(f"checkpoint/{SOURCE_DOMAIN}-{MODEL_CHECKPOINT}/mask", save_function=accelerator.save)
# %%
accelerator.print(f"{os.path.basename(__file__)} finished.")