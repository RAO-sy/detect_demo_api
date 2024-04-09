import flask
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
import json

import argparse
import glob
import os
import numpy as np
import torch

from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic

app = flask.Flask(__name__)
CORS(app, supports_credentials=True)
executor = ThreadPoolExecutor(10)

class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(args.ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')

    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)


def args(reference_model_name, scoring_model_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str,
                        default=reference_model_name)  # use gpt-j-6B for more accurate detection
    parser.add_argument('--scoring_model_name', type=str, default=scoring_model_name)
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--ref_path', type=str, default="./local_infer_ref")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    return args


# 全局定义六种组合模型和对应的API接口
reference_model_name_lis = ['gpt-neo-2.7B','gpt-j-6B','gpt2-xl']
scoring_model_name_lis = ['gpt-neo-2.7B','gpt-j-6B','gpt2-xl']
criterion_fn = get_sampling_discrepancy_analytic


# load three models
class init_neo_model:
    def __init__(self,args):
        self.tokenizer = load_tokenizer('gpt-neo-2.7B', args.dataset, args.cache_dir)
        self.model = load_model('gpt-neo-2.7B', args.device, args.cache_dir, is_half=True)
        self.model.eval()

class init_j_model:
    def __init__(self,args):
        self.tokenizer = load_tokenizer('gpt-j-6B', args.dataset, args.cache_dir)
        self.model = load_model('gpt-j-6B', args.device, args.cache_dir, is_half=True)
        self.model.eval()

class init_xl_model:
    def __init__(self,args):
        self.tokenizer = load_tokenizer('gpt2-xl', args.dataset, args.cache_dir)
        self.model = load_model('gpt2-xl', args.device, args.cache_dir, is_half=True)
        self.model.eval()

# 全局初始化加载，包括模型和数据位置，三个模型的加载，以及评价指标的加载
global_args = args(reference_model_name="gpt-neo-2.7B",scoring_model_name="gpt-neo-2.7B")
neo_model = init_neo_model(global_args)
j_model = init_j_model(global_args)
xl_model = init_xl_model(global_args)
prob_estimator = ProbEstimator(global_args)


def return_data(code, msg, data, cookie="", ToNone=True):
    if ToNone and len(data) <= 0:
        data = None
    jsonStr = {
        'code': code,
        'msg': msg,
        'data': data
    }
    response = flask.make_response(flask.jsonify(jsonStr))
    if cookie:
        for key, value in cookie.items():
            response.set_cookie(key, value, max_age=3600 * 12)
    return response


def evaluate_text(args, text, reference_model, scoring_model, prob_estimator):
    # evaluate text
    tokenized = scoring_model.tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(
        args.device)
    labels = tokenized.input_ids[:,1:]
    with torch.no_grad():
        logits_score = scoring_model.model(**tokenized).logits[:, :-1]
        if args.scoring_model_name == args.reference_model_name:
            logits_ref = logits_score
        else:
            tokenized = reference_model.tokenizer(text, return_tensors="pt", padding=True,
                                                        return_token_type_ids=False).to(args.device)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = reference_model.model(**tokenized).logits[:, :-1]
        crit = criterion_fn(logits_ref, logits_score, labels)
    # estimate the probability of machine generated text
    prob = prob_estimator.crit_to_prob(crit)
    return crit, prob


def process_request(args, text, reference_model, scoring_model, prob_estimator):
    crit, prob = evaluate_text(args, text, reference_model, scoring_model, prob_estimator)
    return crit, prob

# reference_model: gpt-neo-2.7B   scoring_model: gpt-neo-2.7B
args_neo_neo = args(reference_model_name_lis[0],scoring_model_name_lis[0])
@app.route("/detect_neo_neo", methods=["GET", "POST"])
def local_infer_neo_neo():
    info = {}
    if flask.request.method == 'POST':
        try:
            data = flask.request.data.decode('utf-8').replace('"','')
        except:
            info={}
    sentence = data
    data = {"sentence": sentence
            }
    print(data)
    if 'sentence' in data.keys():
        try:
            text = data["sentence"]
            # 使用进程池处理请求
            future = executor.submit(process_request, args_neo_neo, text, neo_model, neo_model, prob_estimator)
            # 获取处理结果
            crit, prob = future.result()
            # crit, prob = evaluate_text(args, text)
            info["crit"], info["prob"] = round(crit,3), round(prob*100,2)
        except:
            return return_data(400, 'Bad request', '')

    return return_data(0, '', info)


# reference_model:gpt-neo-2.7B   scoring_model:gpt-j-6B
args_neo_j = args(reference_model_name_lis[0],scoring_model_name_lis[1])
@app.route("/detect_neo_j", methods=["GET", "POST"])
def local_infer_neo_j():
    info = {}
    if flask.request.method == 'POST':
        try:
            data = flask.request.data.decode('utf-8').replace('"','')
        except:
            info={}
    sentence = data
    data = {"sentence": sentence
            }
    print(data)
    if 'sentence' in data.keys():
        try:
            text = data["sentence"]
            # 使用进程池处理请求
            future = executor.submit(process_request, args_neo_j, text, neo_model, j_model, prob_estimator)
            # 获取处理结果
            crit, prob = future.result()
            # crit, prob = evaluate_text(args, text)
            info["crit"], info["prob"] = round(crit,3), round(prob*100,2)
        except:
            return return_data(400, 'Bad request', '')

    return return_data(0, '', info)

args_neo_xl = args(reference_model_name_lis[0],scoring_model_name_lis[2])
@app.route("/detect_neo_xl", methods=["GET", "POST"])
def local_infer_neo_xl():
    info = {}
    if flask.request.method == 'POST':
        try:
            data = flask.request.data.decode('utf-8').replace('"','')
        except:
            info={}
    sentence = data
    data = {"sentence": sentence
            }
    print(data)
    if 'sentence' in data.keys():
        try:
            text = data["sentence"]
            # 使用进程池处理请求
            future = executor.submit(process_request, args_neo_j, text, neo_model, xl_model, prob_estimator)
            # 获取处理结果
            crit, prob = future.result()
            # crit, prob = evaluate_text(args, text)
            info["crit"], info["prob"] = round(crit,3), round(prob*100,2)
        except:
            return return_data(400, 'Bad request', '')

    return return_data(0, '', info)


args_j_neo = args(reference_model_name_lis[1],scoring_model_name_lis[0])
@app.route("/detect_j_neo", methods=["GET", "POST"])
def local_infer_j_neo():
    info = {}
    if flask.request.method == 'POST':
        try:
            data = flask.request.data.decode('utf-8').replace('"','')
        except:
            info={}
    sentence = data
    data = {"sentence": sentence
            }
    print(data)
    if 'sentence' in data.keys():
        try:
            text = data["sentence"]
            # 使用进程池处理请求
            future = executor.submit(process_request, args_j_neo, text, j_model, neo_model, prob_estimator)
            # 获取处理结果
            crit, prob = future.result()
            # crit, prob = evaluate_text(args, text)
            info["crit"], info["prob"] = round(crit,3), round(prob*100,2)
        except:
            return return_data(400, 'Bad request', '')

    return return_data(0, '', info)

args_j_j = args(reference_model_name_lis[1],scoring_model_name_lis[1])
@app.route("/detect_j_j", methods=["GET", "POST"])
def local_infer_j_j():
    info = {}
    if flask.request.method == 'POST':
        try:
            data = flask.request.data.decode('utf-8').replace('"','')
        except:
            info={}
    sentence = data
    data = {"sentence": sentence
            }
    print(data)
    if 'sentence' in data.keys():
        try:
            text = data["sentence"]
            # 使用进程池处理请求
            future = executor.submit(process_request, args_j_j, text, j_model, j_model, prob_estimator)
            # 获取处理结果
            crit, prob = future.result()
            # crit, prob = evaluate_text(args, text)
            info["crit"], info["prob"] = round(crit,3), round(prob*100,2)
        except:
            return return_data(400, 'Bad request', '')

    return return_data(0, '', info)

args_j_xl = args(reference_model_name_lis[1],scoring_model_name_lis[2])
@app.route("/detect_j_xl", methods=["GET", "POST"])
def local_infer_j_xl():
    info = {}
    if flask.request.method == 'POST':
        try:
            data = flask.request.data.decode('utf-8').replace('"','')
        except:
            info={}
    sentence = data
    data = {"sentence": sentence
            }
    print(data)
    if 'sentence' in data.keys():
        try:
            text = data["sentence"]
            # 使用进程池处理请求
            future = executor.submit(process_request, args_j_xl, text, j_model, xl_model, prob_estimator)
            # 获取处理结果
            crit, prob = future.result()
            # crit, prob = evaluate_text(args, text)
            info["crit"], info["prob"] = round(crit,3), round(prob*100,2)
        except:
            return return_data(400, 'Bad request', '')

    return return_data(0, '', info)

args_xl_neo = args(reference_model_name_lis[2],scoring_model_name_lis[0])
@app.route("/detect_xl_neo", methods=["GET", "POST"])
def local_infer_xl_neo():
    info = {}
    if flask.request.method == 'POST':
        try:
            data = flask.request.data.decode('utf-8').replace('"','')
        except:
            info={}
    sentence = data
    data = {"sentence": sentence
            }
    print(data)
    if 'sentence' in data.keys():
        try:
            text = data["sentence"]
            # 使用进程池处理请求
            future = executor.submit(process_request, args_xl_neo, text, xl_model, neo_model, prob_estimator)
            # 获取处理结果
            crit, prob = future.result()
            # crit, prob = evaluate_text(args, text)
            info["crit"], info["prob"] = round(crit,3), round(prob*100,2)
        except:
            return return_data(400, 'Bad request', '')

    return return_data(0, '', info)

args_xl_j = args(reference_model_name_lis[2],scoring_model_name_lis[1])
@app.route("/detect_xl_j", methods=["GET", "POST"])
def local_infer_xl_j():
    info = {}
    if flask.request.method == 'POST':
        try:
            data = flask.request.data.decode('utf-8').replace('"','')
        except:
            info={}
    sentence = data
    data = {"sentence": sentence
            }
    print(data)
    if 'sentence' in data.keys():
        try:
            text = data["sentence"]
            # 使用进程池处理请求
            future = executor.submit(process_request, args_xl_j, text, xl_model, j_model, prob_estimator)
            # 获取处理结果
            crit, prob = future.result()
            # crit, prob = evaluate_text(args, text)
            info["crit"], info["prob"] = round(crit,3), round(prob*100,2)
        except:
            return return_data(400, 'Bad request', '')

    return return_data(0, '', info)

args_xl_xl = args(reference_model_name_lis[2],scoring_model_name_lis[2])
@app.route("/detect_xl_xl", methods=["GET", "POST"])
def local_infer_xl_xl():
    info = {}
    if flask.request.method == 'POST':
        try:
            data = flask.request.data.decode('utf-8').replace('"','')
        except:
            info={}
    sentence = data
    data = {"sentence": sentence
            }
    print(data)
    if 'sentence' in data.keys():
        try:
            text = data["sentence"]
            # 使用进程池处理请求
            future = executor.submit(process_request, args_xl_xl, text, xl_model, xl_model, prob_estimator)
            # 获取处理结果
            crit, prob = future.result()
            # crit, prob = evaluate_text(args, text)
            info["crit"], info["prob"] = round(crit,3), round(prob*100,2)
        except:
            return return_data(400, 'Bad request', '')

    return return_data(0, '', info)

if __name__ == '__main__':
    app.run(host='10.0.2.37')
    # # app.run()

