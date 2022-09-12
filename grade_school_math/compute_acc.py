import argparse
import glob
import random
from tqdm import tqdm
import sys
import json
import numpy as np
import torch as th
import torch.nn.functional as F
from dataset import get_examples, is_correct
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from calculator import use_calculator


EQUALS_TOKENS = set([28, 796, 47505])


def eval_likelihood(model, tokenizer, qn, ans, device):
    qn = tokenizer([qn], padding=False, return_tensors='pt')['input_ids'].to(device)
    ans = tokenizer([ans], padding=False, return_tensors='pt')['input_ids'].to(device)

    toks = th.cat([qn, ans], dim=1)
    tgts = th.cat([th.full_like(qn, -100), ans], dim=1)

    likelihood = -model(toks, labels=tgts).loss.item()# * ans.shape[1]
    return np.exp(likelihood), ans.shape[1]


def sample(model, qn, tokenizer, device, sample_len=100, calc=True):
    # Samples an answer from the model
    q_len = len(qn)
    for _ in range(sample_len):
        toks = tokenizer([qn], padding=False, return_tensors='pt').to(device)
        orig_len = toks['input_ids'].shape[1]

        out = model.generate(
            **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id
        )
        text = tokenizer.batch_decode(out)[0]

        if calc and out[0, -1].item() in EQUALS_TOKENS:
            answer = use_calculator(text)
            if answer is not None:
                text = text + str(answer) + ">>" 
        
        qn = text         
        
        if out[0, -1].item() == tokenizer.eos_token_id:
            break
    ans = qn[q_len:]
    return ans

def acc(model, tokenizer, dset, device, sample_len=100):
    correct, total = 0, 0
    pbar = tqdm(total=len(dset))

    info = []
    for ex in dset:
        qn = ex['question']
        ans_pred = sample(model, qn, tokenizer, device, sample_len)
        likeli, n_tokens = eval_likelihood(model, tokenizer, qn, ans_pred, device)
        if is_correct(ans_pred, ex):
            correct += 1
        total += 1
        info.append((qn, ex['answer'], ans_pred, n_tokens, likeli, is_correct(ans_pred, ex)))
        pbar.update(1)
        pbar.set_description(f'Accuracy: {100. * correct / total:.1f}')

    return info

    
def main():
    device = th.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print('Model', args.id)

    model_paths = glob.glob('model_ckpts/*')
    model = GPT2LMHeadModel.from_pretrained(model_paths[args.id]).eval().to(device)
    print("Model Loaded")

    th.set_grad_enabled(False)
    
    if args.dataset == 'train': 
        examples = get_examples('train')
    elif args.dataset == 'test':
        examples = get_examples('test')
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')
    if args.n is not None:
        examples = examples[:args.n]

    sample_len = 400
    info = acc(model, tokenizer, examples, device, sample_len)
    json.dump(info, open(f'results/model_{args.id}_results.json', 'w'))

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='train')
    parser.add_argument('-i', '--id', type=int, default=0)
    parser.add_argument('-n', '--n', type=int, default=None)
    args = parser.parse_args()
    main()
