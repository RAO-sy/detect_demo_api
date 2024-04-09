# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import time
import numpy as np
import torch
import math
import torch.nn.functional as F
import tqdm
import argparse
import json
from data_builder import load_data
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics
import custom_datasets

class ContextSampler:
    def __init__(self, args):
        self.args = args
        self.tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        self.model = load_model(args.reference_model_name, args.device, args.cache_dir)
        self.model.eval()

    def _score(self, input_ids, labels):
        with torch.no_grad():
            logits = self.model(input_ids).logits
            lprobs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1))
            return lprobs.squeeze(-1)

    def _sample_from_model(self, text):
        all_encoded = self.tokenizer(text, return_tensors="pt", padding=True).to(self.args.device)
        input_ids = all_encoded['input_ids'][..., :-1]
        labels = all_encoded['input_ids'][..., 1:]
        all_lprobs = self._score(input_ids, labels)[0]
        # prompt tokens

        start = self.args.prompt_tokens
        if self.args.dataset == 'pubmed':
            context = text[:text.index(custom_datasets.SEPARATOR)]
            all_encoded = self.tokenizer(context, return_tensors="pt", padding=True)
            start = all_encoded['input_ids'].size(-1)
        # interval since start
        result = {'text': text, 'start': start}
        intervals = []
        interval_tokens = math.ceil((input_ids.size(-1) - start)*self.args.interval_tokens_ratio)

        while start < input_ids.size(-1):
            end = min(start + interval_tokens, input_ids.size(-1))
            context_ids = input_ids[:, :start].repeat_interleave(self.args.nsamples, dim=0)

            sampling_kwargs = {'temperature': self.args.temperature}
            if self.args.do_top_p:
                sampling_kwargs['top_p'] = self.args.top_p
            elif self.args.do_top_k:
                sampling_kwargs['top_k'] = self.args.top_k
            expect_length = end - start
            outputs = self.model.generate(input_ids=context_ids, min_new_tokens=expect_length - 1, max_new_tokens=expect_length,
                                        do_sample=True, **sampling_kwargs, pad_token_id=self.tokenizer.eos_token_id,
                                          eos_token_id=self.tokenizer.eos_token_id)
            lprobs = self._score(outputs[..., :-1], outputs[..., 1:])
            lprobs = lprobs[..., start-1:].sum(dim=-1).data.cpu().numpy().tolist()
            samples = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            intervals.append({'start': start, 'end': end, 'text_lprob': all_lprobs[start: end].sum().item(),
                              'samples': samples, 'sample_lprobs': lprobs})
            start = end
        result['intervals'] = intervals
        return result

    def generate_samples(self, text):
        return self._sample_from_model(text)

def get_general_sampling_discrepancy(samples):
    ntokens = 0
    lprob0 = 0.0
    lprob1_mean = 0.0
    lprob1_var = 0.0
    for item in samples['intervals']:
        ntokens += item['end'] - item['start']
        lprob0 += item['text_lprob']
        lprob1_mean += np.mean(item['sample_lprobs'])
        lprob1_var += np.var(item['sample_lprobs'])
    lprob0 /= ntokens
    lprob1_mean /= ntokens
    lprob1_var /= ntokens
    discrepancy = (lprob0 - lprob1_mean) / np.sqrt(lprob1_var)
    return discrepancy


def experiment(args):
    # load model
    sampler = ContextSampler(args)
    # load data
    data = load_data(args.dataset_file)
    # n_samples = len(data["sampled"])
    n_samples = 1
    # evaluate criterion
    name = "general_sampling_discrepancy"
    criterion_fn = get_general_sampling_discrepancy
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    start_time = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        human_text = data["original"][idx]
        machine_text = data["sampled"][idx]
        # original text
        human_samples = sampler.generate_samples(human_text)
        human_crit = criterion_fn(human_samples)
        # sampled text
        machine_samples = sampler.generate_samples(machine_text)
        machine_crit = criterion_fn(machine_samples)
        # result
        results.append({"original": human_text,
                        "original_crit": human_crit,
                        "sampled": machine_text,
                        "sampled_crit": machine_crit})
    end_time = time.time()
    run_time = end_time - start_time
    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc,
                'cost_time':run_time}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="../exp_main/results/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="../exp_main/data/xsum_gpt2-xl")
    parser.add_argument('--reference_model_name', type=str, default="gpt2")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2")

    parser.add_argument('--nsamples', type=int, default=20)
    parser.add_argument('--prompt_tokens', type=int, default=30)
    parser.add_argument('--interval_tokens_ratio', type=float, default=1)

    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../../cache")
    args = parser.parse_args()

    experiment(args)
