import torch as th
import torch.nn.functional as F
from dataset import get_examples
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
    ans = qn[q_len:]
    return ans

    
def eval_likelihood(model, tokenizer, qn, ans, device):
    qn = tokenizer([qn], padding=False, return_tensors='pt')['input_ids'].to(device)
    ans = tokenizer([ans], padding=False, return_tensors='pt')['input_ids'].to(device)
    toks = th.cat([qn, ans], dim=1)
    tgts = th.cat([th.full_like(qn, -100), ans], dim=1)

    nll = model(toks, labels=tgts).loss
    return nll


def main():
    device = th.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("model_ckpts")
    model.eval()
    model.to(device)
    print("Model Loaded")

    th.set_grad_enabled(False)
    
    train_examples, test_examples = get_examples('train'), get_examples('test')
    qn = test_examples[1]['question']
    sample_len = 100
    print(qn.strip())

    ans = sample(model, qn, tokenizer, device)
    print('ans:', ans)
    print('likelihood:', eval_likelihood(model, tokenizer, qn, ans, device))


if __name__ == '__main__':
    main()