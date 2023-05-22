# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from src.utils import extract_id, read_jsonl, read_wikipedia_links
import csv

from llama import LLaMA, ModelArgs, Tokenizer, Transformer

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\nGiven a sentence, output the Wikipedia link for a mention that is surrounded by the special tokens [START_ENT] and [END_ENT].\n\n### Input:\n{input}\n\n### Output:"
    ),
}


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(adapter_checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def extract_output(text):
    try:
        split_text = text.split('### Output:')
        return split_text[1].strip()  # strip is used to remove leading/trailing white spaces
    except IndexError:
        return "Output section not found in the text"
    
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    temperature: float = 0.1,
    top_p: float = 0.75,
    max_seq_len: int = 1200,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size)

    data = read_jsonl("data/aidayago2-test_without_answers-kilt.jsonl")

    with open("results.csv", "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["id", "idx","text", "prediction"])

        idx_cnt = 0
        prev_id = data[0]["id"]
        for i in range(0, len(data), 10):
            batch = data[i:i+10]    
            inputs = [d["input"] for d in batch]
            ids = [extract_id(d["id"]) for d in batch]
            idx_cnts = []

            for id in ids:
                if id == prev_id:
                    idx_cnt += 1
                else:
                    prev_id = id
                    idx_cnt = 0
                idx_cnts.append(idx_cnt)
                
            prompts = [PROMPT_DICT["prompt_input"].format_map({"input": x}) for x in inputs]

            results = generator.generate(prompts, max_gen_len=512, temperature=temperature, top_p=top_p)
            for id, idx_cnt, text, result in zip(ids, idx_cnts, inputs, results):
                out = extract_output(result)
                csv_writer.writerow([id, idx_cnt, text, out])

            # break


if __name__ == "__main__":
    fire.Fire(main)
