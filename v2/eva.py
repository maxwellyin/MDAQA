# evaluation on target domain
# %%
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
from util import preprocess_validation_examples, compute_metrics, SEED, SOURCE_DOMAIN, TARGET_DOMAIN, MODEL_CHECKPOINT, BATCH_SIZE
import roberta
# %%
TRAIN_DATA = f"../data/{TARGET_DOMAIN}_train"
TEST_DATA = f"../data/{TARGET_DOMAIN}_test"
VAL_SPLIT = "test"
# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
test_dataset = load_from_disk(TEST_DATA).shuffle(seed=SEED)
raw_datasets = DatasetDict({'test': test_dataset})
# %%
# model_checkpoint = f"checkpoint/{SOURCE_DOMAIN}-{MODEL_CHECKPOINT}/self-train/8"
model_checkpoint = f"checkpoint/{SOURCE_DOMAIN}-{MODEL_CHECKPOINT}/BMW_source"
# %%
validation_dataset = raw_datasets[VAL_SPLIT].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets[VAL_SPLIT].column_names,
)
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")
# %%
model = roberta.RobertaBMW.from_pretrained(model_checkpoint)
# %%
eval_dataloader = DataLoader(
    validation_set, collate_fn=default_data_collator, batch_size=BATCH_SIZE
)
# %%
accelerator = Accelerator()
model, eval_dataloader = accelerator.prepare(
    model, eval_dataloader
)
# %%
model.eval()
start_logits = []
end_logits = []
accelerator.print("Evaluation!")
for batch in tqdm(eval_dataloader):
    with torch.no_grad():
        outputs = model(**batch)

    start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
    end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

start_logits = np.concatenate(start_logits)
end_logits = np.concatenate(end_logits)
start_logits = start_logits[: len(validation_dataset)]
end_logits = end_logits[: len(validation_dataset)]

metrics = compute_metrics(
    start_logits, end_logits, validation_dataset, raw_datasets[VAL_SPLIT]
)
accelerator.print(metrics)
# %%