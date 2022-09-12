import argparse
import glob
import random
from tqdm import tqdm
import json
import numpy as np
import torch as th
import torch.nn.functional as F
from dataset import get_examples, get_squad, is_correct
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from calculator import use_calculator


EQUALS_TOKENS = set([28, 796, 47505])


def sample(model, qn, tokenizer, device, sample_len=100, calc=True):
    # Samples an answer from the model
    q_len = len(qn)
    for _ in range(sample_len):
        toks = tokenizer([qn], padding=False, return_tensors='pt').to(device)
        orig_len = toks['input_ids'].shape[1]

        out = model.generate(
            **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id,
            do_sample=True
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

def sample_dataset(model, tokenizer, dset, device, sample_len=100, calc=True):
    answers = []
    for ex in dset:
        qn = ex['question']
        ans = sample(model, qn, tokenizer, device, sample_len, calc=calc)
        if calc:
            c = is_correct(ans, ex)
        else:
            c = False
        answers.append((ans, c))
    return answers

    
def eval_likelihood(model, tokenizer, qn, ans, device):
    qn = tokenizer([qn], padding=False, return_tensors='pt')['input_ids'].to(device)
    ans = tokenizer([ans], padding=False, return_tensors='pt')['input_ids'].to(device)

    toks = th.cat([qn, ans], dim=1)
    tgts = th.cat([th.full_like(qn, -100), ans], dim=1)

    likelihood = -model(toks, labels=tgts).loss.item()# * ans.shape[1]
    return np.exp(likelihood)


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
    n_models = len(models)
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

    sample_len = 400

    answers = [None] * n_models
    for i, model in enumerate(tqdm(models)):
        model.to(device)
        ans = sample_dataset(model, tokenizer, examples, device, sample_len, calc=args.dataset != 'squad')
        answers[i] = ans
        model.cpu()

    json.dump(answers, open(f'answers_{args.dataset}.json', 'w'))

    likelihoods = np.zeros((args.n, n_models, n_models), dtype=np.float64)
    pbar = tqdm(total=n_models ** 2 * args.n)
    for i in range(n_models): # prediction of ith model
        for j in range(n_models): # applied to likelihood of jth model
            model = models[j].to(device)
            for k in range(args.n): # data point k
                ex = examples[k]
                ans = answers[i][k][0]
                likelihood = eval_likelihood(model, tokenizer, ex['question'], ans, device)
                likelihoods[k, i, j] = likelihood
                pbar.update(1)
            model.cpu()

    np.save(f'ensemble_matrices_{args.dataset}.npy', likelihoods)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='test')
    parser.add_argument('-n', '--n', type=int, default=100)
    args = parser.parse_args()
    main()
