from flask import Flask, request
import json
import time
import argparse
import random

import numpy as np
import torch

import sentencepiece as spm
from dialogue_generation_models.configuration_meena import MeenaConfig
from dialogue_generation_models.modeling_meena import MeenaForConditionalGeneration

app = Flask(__name__)
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-model-path", type=str, help="Path to pre-trained model", required=True)
parser.add_argument(
  "--model-config-path",
  default="./configs/base_meena_config.json",
  type=str,
  help="Path to model configuration file",
)
parser.add_argument(
  "--tokenizer-model-path",
  default="./tokenizer/kr_spm.model",
  type=str,
  help="Path to Sentencepiece model",
)
parser.add_argument(
  "--decoding-method",
  default="top_p",
  type=str,
  help="Decoding method (beam_search or top_p)",
)


def main(args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()

  np.random.seed(42)
  torch.manual_seed(42)
  random.seed(42)
  if n_gpu > 0:
      torch.cuda.manual_seed_all(42)

  tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_model_path)

  config = MeenaConfig.from_json(args.model_config_path)
  model = MeenaForConditionalGeneration(config)
  model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"))
  model.eval()
  model.to(device)

  print("모델 로드 완료 : ", time.time() - start)
  def modelrun(contexts):
      for context in contexts:
          print('모델 요청', contexts)
          input_ids = (
              torch.tensor(
                  [
                      token_id
                      for utterance in context
                      for token_id in tokenizer.encode(utterance, out_type=int) + [config.sept_token_id]
                  ]
                  + [config.bos_token_id]
              )
              .unsqueeze(0)
              .to(device)
          )

          input_ids = input_ids

          if args.decoding_method == "top_p":
              outputs = model.generate(
                  input_ids=input_ids,
                  max_length=48,
                  min_length=8,
                  temperature=1.0,
                  do_sample=True,
                  top_p=0.8,
                  pad_token_id=config.pad_token_id,
                  bos_token_id=config.bos_token_id,
                  eos_token_id=config.eos_token_id,
                  repetition_penalty=1.3,
                  no_repeat_ngram_size=3,
                  num_return_sequences=10,
              )
          elif args.decoding_method == "beam_search":
              outputs = model.generate(
                  input_ids=input_ids,
                  max_length=48,
                  min_length=8,
                  num_beams=10,
                  pad_token_id=config.pad_token_id,
                  bos_token_id=config.bos_token_id,
                  eos_token_id=config.eos_token_id,
                  repetition_penalty=1.3,
                  no_repeat_ngram_size=3,
                  num_return_sequences=10,
              )
          else:
              raise ValueError("Enter the right decoding method (top_p or beam_search)")

          result = []
          for output in outputs.tolist():
              result.append(tokenizer.decode(output))
          
          return result
          
  @app.route("/run", methods=['POST'])
  def runbypost():
    return(json.dumps(modelrun(request.get_json()), ensure_ascii=False))

if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
  app.run(debug=True)
