import argparse
import os
import os.path as osp
import torch as th
import torch.nn.functional as F
from dataset import get_examples, GSMDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def main():
    th.manual_seed(args.id)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_examples = get_examples("train")
    train_dset = GSMDataset(tokenizer, train_examples)

    device = th.device("cuda")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model.to(device)
    model.train()

    config = GPT2Config()
    rnd_prior = GPT2LMHeadModel(config=config)
    rnd_prior.to(device)
    rnd_prior.eval()

    train_loader = DataLoader(train_dset, batch_size=16, shuffle=True)
    optim = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 20
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = F.log_softmax(model(**batch).logits)
            rnd_logits = F.log_softmax(rnd_prior(**batch).logits)

            logits = logits + rnd_logits
            labels = batch['input_ids']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:]
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.shape[-1])
            loss = loss + args.reg_weight * F.mse_loss(logits, rnd_logits)
            
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")


    os.makedirs(args.output_dir)    
    model.save_pretrained(osp.join(args.output_dir, f"{args.id}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id', type=int, default=0)
    parser.add_argument('-o', '--output_dir', type=str, default='model_ckpts')
    parser.add_argument('-w', '--reg_weight', type=float, default=0.)
    args = parser.parse_args()
    print(args)
    main()
