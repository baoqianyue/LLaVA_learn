import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from PIL import Image
import math
import torch.nn.functional as F
import transformers
# MemVR
from memvr import apply_memvr_llama, LlamaMLP, reset_surgery


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print('prompt', prompt)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    device= args.cuda_device
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # MemVR init
    transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP

    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, args.model_base, model_name, device_map=device)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    
    # MemVR
    if args.apply_memvr == 'memvr':
        apply_memvr_llama(
            self=model,
            starting_layer=args.starting_layer,
            ending_layer=args.ending_layer,
            entropy_threshold=args.entropy_threshold,
            retracing_ratio=args.retracing_ratio,
            device=device,
        )
    # print('model parameters:', model.model.trainable_params)
    model_parameters = sum(p.numel() for p in model.model.trainable_params if p.requires_grad)
    print(f"Model parameters: {model_parameters / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.model.trainable_params, lr=args.rl_lr, eps=1e-06, weight_decay=args.rl_weight_decay)

    # for i, p in enumerate(model.model.trainable_params):
    #     print(f"Param {i} requires_grad: {p.requires_grad}, shape: {p.shape}")

    optim_state = deepcopy(optimizer.state_dict())
    # scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        input_ids = input_ids.to(device=device, non_blocking=True)

        for step in range(args.rl_steps):
            model.model.reward_list = []
            with torch.no_grad():
                model.eval()
                
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device=device, non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    num_return_sequences=args.num_beams,
                )
                
                reward_list = torch.stack(model.model.reward_list, dim=0)
                rewards = reward_list.mean(dim=0)

            # print('reward_list', reward_list)

            model.train()
            optimizer.zero_grad()

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return_tokens = tokenizer(outputs, return_tensors="pt", padding=True).to(device)
            tokens = return_tokens.input_ids
            attention_mask = return_tokens.attention_mask
            # print('tokens', tokens.shape)

            # torch.cuda.empty_cache()
            
            # with torch.cuda.amp.autocast():
            rl_outputs = model(tokens, attention_mask)
            rl_logits = rl_outputs.logits
            # print('rl_logits', rl_logits.shape)

            all_loss = F.cross_entropy(rl_logits.reshape(-1, rl_logits.shape[-1]), tokens.flatten(),
                                                ignore_index=0, reduction='none').reshape(rl_logits.shape[0], -1)
                
            loss = torch.mean(rewards * all_loss.mean(dim=-1))
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()

            
        # print('outputs', outputs)
        model.eval()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device=device, non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1,
                use_cache=True,
                num_return_sequences=1,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


        print("\n Outputs: ", outputs)
        reset_surgery(model)
        optimizer.load_state_dict(optim_state)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/data/eval/pope/val2014")
    parser.add_argument("--question-file", type=str, default="./playground/data/eval/pope/llava_pope_test.jsonl")
    parser.add_argument("--answers-file", type=str, default="./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1)

    # MemVR
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--vision-retracing", type=str, default="default")
    parser.add_argument("--retracing-ratio", type=float, default=0.0)
    parser.add_argument("--entropy-threshold", type=float, default=0.75)
    parser.add_argument("--starting-layer", type=int, default=5)
    parser.add_argument("--ending-layer", type=int, default=16)
    parser.add_argument("--apply-memvr", type=str, default='default')

    # RL
    parser.add_argument("--rl_steps", type=int, default=3)
    parser.add_argument("--rl_lr", type=float, default=1e-5)
    parser.add_argument("--rl_weight_decay", type=float, default=5e-4)
    args = parser.parse_args()

    eval_model(args)
