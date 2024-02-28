# change learning rate
# %%
import torch, shutil
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, pipeline, default_data_collator, get_scheduler
from util import preprocess_training_examples, preprocess_validation_examples, compute_metrics, BATCH_SIZE, SOURCE_DOMAIN, TARGET_DOMAIN, MODEL_CHECKPOINT, SEED, NUM_LOOPS, BOTTLE_DIM
import os
from accelerate import Accelerator
import roberta
# %%
TRAIN_DATA = f"../data/{TARGET_DOMAIN}_train"
TEST_DATA = f"../data/{TARGET_DOMAIN}_test"
VAL_SPLIT = "test"
THRESHOLD = 0.7
LR_BASE = 2e-5
LR_BOTTLE = 5e-3
MASK_DISTANCE = 3 # default: 3
checkpoint_dir = f"checkpoint/{SOURCE_DOMAIN}-{MODEL_CHECKPOINT}/self_train"
model_checkpoint = f"checkpoint/{SOURCE_DOMAIN}-{MODEL_CHECKPOINT}/BMW_source"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = roberta.RobertaBMW.from_pretrained(model_checkpoint)
source_mask = model.getMask().detach().cuda()
# %%
train_datasets = load_from_disk(TRAIN_DATA).shuffle(seed=SEED).select(range(2000))
train_datasets2 = train_datasets.train_test_split(train_size=.5, seed=SEED)
test_dataset = load_from_disk(TEST_DATA).shuffle(seed=SEED)
raw_datasets = DatasetDict({
    'train': train_datasets2['train'],
    'validation': train_datasets2['test'],
    'test': test_dataset})
# %%
validation_dataset = raw_datasets[VAL_SPLIT].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets[VAL_SPLIT].column_names,
)
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")
# %%
def train(train_dataset, model, masks_old):
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=BATCH_SIZE,
    )

    params_base = [v for k, v in model.named_parameters() if "bottleMask" not in k and "qa_outputs" not in k]
    params_m = [v for k, v in model.named_parameters() if "bottleMask" in k or "qa_outputs" in k]
    optimizer_base = AdamW(params_base, lr= LR_BASE)
    optimizer_m = AdamW(params_m, lr=LR_BOTTLE)

    accelerator = Accelerator()
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

    lr_scheduler_m = get_scheduler(
        "linear",
        optimizer=optimizer_m,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_train_epochs):       
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            target_mask = model.getMask()
            loss = outputs.loss
            accelerator.backward(loss)

            for n, p in model.bottleMask.bottleneck.named_parameters():
                if n.find('weight') != -1:
                    mask_ = ((1 - masks_old)).view(-1, 1).expand(BOTTLE_DIM, 768).cuda()
                    p.grad.data *= mask_
                else:
                    mask_ = ((1 - masks_old)).squeeze().cuda()
                    p.grad.data *= mask_

            for n, p in model.qa_outputs.named_parameters():
                if n.find('weight_v') != -1:
                    masks__=masks_old.view(1,-1).expand(2,BOTTLE_DIM)
                    mask_ = ((1 - masks__)).cuda()
                    p.grad.data *= mask_

            if (abs(target_mask - masks_old)).sum() > MASK_DISTANCE:
                model.bottleMask.m.grad.data = torch.zeros(model.bottleMask.m.shape).cuda()

            optimizer_base.step()
            lr_scheduler_base.step()
            optimizer_base.zero_grad()

            optimizer_m.step()
            lr_scheduler_m.step()
            optimizer_m.zero_grad()

            progress_bar.update(1)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    return unwrapped_model
# %%
def eval(validation_set, model):
    eval_dataloader = DataLoader(
        validation_set, collate_fn=default_data_collator, batch_size=BATCH_SIZE
    )
    accelerator = Accelerator()
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

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
    return metrics
# %%
def main(checkpoint_dir: str, i:int, model, source_mask):
    question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer, device=torch.cuda.is_available()-1)

    def create_pseudo_label(example):
        out = question_answerer(question=example['question'], context=example['context'])
        answers = {'text': [out['answer']], 'answer_start': [out['start']]}
        return {'answers': answers, 'score': out['score']}

    pseudo_set = raw_datasets['train'].map(create_pseudo_label, load_from_cache_file=False)
    pseudo_set2 = pseudo_set.filter(lambda example: example['score']>THRESHOLD)

    train_dataset = pseudo_set2.map(
        preprocess_training_examples,
        batched=True,
        remove_columns=pseudo_set2.column_names,
    )

    train_dataset.set_format("torch")
    model = train(train_dataset, model, source_mask)

    outcome = eval(validation_set, model)
    target_mask = model.getMask().detach().cuda()
    mask_diff = (abs(target_mask - source_mask)).sum().item()
    outcome['mask_diff'] = mask_diff
    # print(outcome)
    return outcome, model
# %%
outcomes = []
for i in range(NUM_LOOPS):
    outcome, model = main(checkpoint_dir, i, model, source_mask)
    outcomes.append(outcome)
    model.save_pretrained(f"checkpoint/{SOURCE_DOMAIN}-{MODEL_CHECKPOINT}/self-train/{i}")
for outcome in outcomes:
    print(outcome)
# %%
print(os.path.basename(__file__))