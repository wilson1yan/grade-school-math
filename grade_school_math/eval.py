import argparse
import glob
import random
from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn.functional as F
from dataset import get_examples, get_squad
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def sample(model, qn, tokenizer, device, sample_len=100):
    # Samples an answer from the model
    q_len = len(qn)
    for _ in range(sample_len):
        toks = tokenizer([qn], padding=False, return_tensors='pt').to(device)
        orig_len = toks['input_ids'].shape[1]

        out = model.generate(
            **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id
        )
        text = tokenizer.batch_decode(out)[0]
        qn = text         
        
        if out[0, -1].item() == tokenizer.eos_token_id:
            break

    ans = qn[q_len:]
    import ipdb; ipdb.set_trace()
    return ans

def sample_dataset(model, tokenizer, dset, device, sample_len=100):
    info = []
    for ex in dset:
        qn = ex['question']
        ans = sample(model, qn, tokenizer, device, sample_len)
        likeli = eval_likelihood(model, tokenizer, qn, ans, device)
        info.append((ans, likeli))
    return info

    
def eval_likelihood(model, tokenizer, qn, ans, device):
    qn = tokenizer([qn], padding=False, return_tensors='pt')['input_ids'].to(device)
    ans = tokenizer([ans], padding=False, return_tensors='pt')['input_ids'].to(device)

    toks = th.cat([qn, ans], dim=1)
    tgts = th.cat([th.full_like(qn, -100), ans], dim=1)

    likelihood = -model(toks, labels=tgts).loss.item()# * ans.shape[1]
    return likelihood


def get_mode_answer(answers):
    # [(ans, likelihood), (ans, likelihood), ...]
    answers = [a[0] for a in answers]
    counts = [answers.count(a) for a in answers]
    max_count = max(counts)
    mode_answers = [a for a, c in zip(answers, counts) if c == max_count]
    return random.choice(mode_answers)


def main():
    device = th.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model_paths = glob.glob('model_ckpts/*')
    models = [GPT2LMHeadModel.from_pretrained(path).eval().cpu()
              for path in model_paths]         
    print("Models Loaded")

    th.set_grad_enabled(False)
    
    if args.dataset == 'train': 
        examples = get_examples('train')
    elif args.dataset == 'test':
        examples = get_examples('test')
    elif args.dataset == 'squad':
        examples = get_squad()
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')
    examples = examples[:args.n]

    sample_len = 100

    samples = []
    for model in tqdm(models):
        model.to(device)
        info = sample_dataset(model, tokenizer, examples, device, sample_len)
        samples.append(info)
        model.cpu()

    samples = list(zip(*samples)) 
    mode_answers = [get_mode_answer(s) for s in samples]

    assert len(mode_answers) == len(examples)
    likelihoods = []
    for model in tqdm(models):
        model.to(device)
        liks = [eval_likelihood(model, tokenizer, ex['question'], ans, device)
                for ex, ans in zip(examples, mode_answers)]
        likelihoods.append(liks)
        model.cpu()
    likelihoods = np.array(likelihoods)
    probs = np.exp(likelihoods)
    mean, std = probs.mean(axis=0), probs.std(axis=0)
    np.savez(f'{args.dataset}_stats.npz', mean=mean, std=std)

    print(mean[:20])
    print(std[:20])
    mean, std = mean.mean(), std.mean()
    print(f'{args.dataset}: {mean} +/- {std}')

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='train')
    parser.add_argument('-n', '--n', type=int, default=100)
    args = parser.parse_args()
    main()
