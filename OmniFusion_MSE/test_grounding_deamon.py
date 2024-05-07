import argparse
import json
import re
import os
import time

from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import get_collate_function, get_dataset
from models import CLIPVisionTower
import warnings
import jsonlines
warnings.filterwarnings("ignore")
torch.inference_mode()

DEVICE = "cuda:0"
PROMPT = "This is a dialog with AI assistant.\n"


def get_gen_params(tokenizer):
    bad_words_ids = tokenizer(["\n", "</s>", ":"], add_special_tokens=False).input_ids + [[13]]
    gen_params = {
        "do_sample": False,
        "max_new_tokens": 50,
        "early_stopping": True,
        "num_beams": 3,
        "repetition_penalty": 1.0,
        "remove_invalid_values": True,
        "eos_token_id": 2,
        "pad_token_id": 2,
        "forced_eos_token_id": 2,
        "use_cache": True,
        "no_repeat_ngram_size": 4,
        "bad_words_ids": bad_words_ids,
        "num_return_sequences": 1,
        "temperature": 0
    }
    return gen_params


def gen_answer(model, tokenizer, clip, projection, grounding_head, query, special_embs, image=None, device=DEVICE):
    bad_words_ids = tokenizer(["\n", "</s>", ":"], add_special_tokens=False).input_ids + [[13]]

    with torch.no_grad():
        image_features = clip.image_processor(image.resize((336, 336)), return_tensors='pt', do_center_crop=False)
        image_embedding = clip(image_features['pixel_values']).to(device=device, dtype=torch.bfloat16)

        projected_vision_embeddings = projection(image_embedding).to(device=device, dtype=torch.bfloat16)
        prompt_ids = tokenizer.encode(f"{PROMPT}", add_special_tokens=False, return_tensors="pt").to(device=device)
        question_ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(device=device)

        prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
        question_embeddings = model.model.embed_tokens(question_ids).to(torch.bfloat16)

        embeddings = torch.cat(
            [
                prompt_embeddings,
                special_embs['SOI'][None, None, ...],
                projected_vision_embeddings,
                special_embs['EOI'][None, None, ...],
                special_embs['USER'][None, None, ...],
                question_embeddings,
                special_embs['BOT'][None, None, ...]
            ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=device)
        # out = model.generate(inputs_embeds=embeddings, **gen_params)
        emb = model(inputs_embeds=embeddings, output_hidden_states=True).get("hidden_states")[-1]
        emb = emb[:, -1, :].to(dtype=torch.bfloat16)
        predictions = grounding_head(emb)[0]
    # generated_texts = tokenizer.batch_decode(out)[0]
    return predictions

def correct_bbox(bbox):
    left, top, right, bottom = bbox
    
    new_bottom = min(max(top, bottom), 1)
    new_top = max(min(top, bottom), 0)
    
    new_left = max(min(left, right), 0)
    new_right = min(max(left, right), 1)
    
    return new_left, new_top, new_right, new_bottom

def get_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    x = max(x1, x3)
    y = max(y1, y3)
    w = min(x2, x4) - x
    h = min(y2, y4) - y
    if w < 0 or h < 0:
        return 0
    return w * h / ((x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - w * h)

def rel_to_abs(bbox, width, height):
    left = width * bbox[0]
    top = height * bbox[1]
    right = width * bbox[2]
    bottom = height * bbox[3]
    
    return left, top, right, bottom

def get_bbox(ans):
    matches = re.search(r'\[(.*?)\]', ans)
    if matches:
        bbox = list(map(float, matches.group(1).split(',')))
        bbox = correct_bbox(bbox)
        return bbox
    else:
        return [0, 0, 0, 0]
    
def get_axis_iou(min_a, max_a, min_b, max_b):
    """ Calculate the Intersection over Union (IoU) for one dimension (x or y). """
    intersection = max(0, min(max_a, max_b) - max(min_a, min_b))
    union = max(max_a, max_b) - min(min_a, min_b)
    if union == 0:
        return 0
    return intersection / union

def get_precision_recall_f1(bbox_pred, bbox_gt):
    """ Calculate precision, recall, and F1-score for predicted bbox against ground truth. """
    # Calculate intersection
    xi1 = max(bbox_pred[0], bbox_gt[0])
    yi1 = max(bbox_pred[1], bbox_gt[1])
    xi2 = min(bbox_pred[2], bbox_gt[2])
    yi2 = min(bbox_pred[3], bbox_gt[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union
    pred_area = (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1])
    gt_area = (bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1])
    union_area = pred_area + gt_area - inter_area

    # Calculate precision, recall, and F1-score
    precision = inter_area / pred_area if pred_area != 0 else 0
    recall = inter_area / gt_area if gt_area != 0 else 0
    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score

def calculate_metrics(bbox_pred, bbox_gt):
    """ Calculate all metrics for bounding boxes. """
    iou = get_iou(bbox_pred, bbox_gt)
    iou_x = get_axis_iou(bbox_pred[0], bbox_pred[2], bbox_gt[0], bbox_gt[2])
    iou_y = get_axis_iou(bbox_pred[1], bbox_pred[3], bbox_gt[1], bbox_gt[3])
    precision, recall, f1_score = get_precision_recall_f1(bbox_pred, bbox_gt)

    return {
        "IoU": iou,
        "IoU X-axis": iou_x,
        "IoU Y-axis": iou_y,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }

def evaluate_ckpt(args, data, ckpt_path, model):
    # Загрузка модели
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, subfolder=args.subfolder, use_fast=False)
    # model = AutoModelForCausalLM.from_pretrained(args.orig_ckpt, subfolder="tuned-model", torch_dtype=torch.bfloat16, device_map=DEVICE)
    # model = AutoModelForCausalLM.from_pretrained(os.path.join(args.ckpt_path, "tuned-model"), torch_dtype=torch.bfloat16, device_map=DEVICE)
    projection = torch.load(os.path.join(ckpt_path, "projection.pt"), map_location=DEVICE)
    special_embs = torch.load(os.path.join(ckpt_path,"special_embeddings.pt"), map_location=DEVICE)
    grounding_head = torch.load(os.path.join(ckpt_path, "grounding_head.pt"), map_location=DEVICE)
    grounding_head.to(dtype=torch.bfloat16)
    
    clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    clip.load_model()
    clip = clip.to(device=DEVICE, dtype=torch.bfloat16)
    torch.inference_mode()
    # model.eval()
    projection.eval()
    clip.eval()
    
    metrics = []
    for sample_id in tqdm(range(args.num_samples)):
        try:
            sample = data[sample_id]
            question = sample["conversations"][0]["value"]
            gt_ans = sample["conversations"][1]["value"]
            img_path = os.path.join(args.data_path, os.path.join(sample["image"]))
            img = Image.open(img_path)

            answer = gen_answer(
                model,
                tokenizer,
                clip,
                projection,
                grounding_head,
                query=question,
                special_embs=special_embs,
                image=img
            )

            
            bbox_gt = sample.get("coords", get_bbox(gt_ans))
            answer = str(list(np.array(answer.to(dtype=torch.float32).cpu())))
            bbox_ans = correct_bbox(get_bbox(answer))
            # bbox_ans = get_bbox(answer)
            
            # Получение размеров изображения
            # width, height = img.size
            # bbox_ans_coors = rel_to_abs(bbox_ans, width, height)
            # bbox_gt_coorts = rel_to_abs(bbox_gt, width, height)

            metrics.append(calculate_metrics(bbox_ans, bbox_gt))
        except ValueError as ex:
            warnings.warn(f"Incorrect output format. Sample id: {sample_id}; answer: {answer}; error: {ex} ")
            metrics.append(calculate_metrics([0, 0, 0, 0], bbox_gt))
    
    mean_metrics = {k: np.sum([m[k] for m in metrics])/args.num_samples for k in metrics[0].keys()}
    
    return mean_metrics
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=200)
    # parser.add_argument('--ckpt_path', type=str, default="./ckpts/Grouning_0/version_2/49999")
    parser.add_argument('--exp_path', type=str, default="./ckpts/Grouning_MSE/version_1/")
    parser.add_argument('--orig_ckpt', type=str, default="./OmniFusion/MicroOmnic")
    parser.add_argument('--data_path', type=str, default="./data/")
    parser.add_argument('--dataset', type=str, default="CWB_flickr30k_test_short_ref.json")
    parser.add_argument('--tokenizer', type=str, default="AIRI-Institute/OmniFusion") # could be taken from exp
    parser.add_argument('--save_file', type=str, default="./deamon_metrics.json")
    parser.add_argument('--check_interval', type=int, default=30)
    args = parser.parse_args()

    if args.tokenizer == "AIRI-Institute/OmniFusion":
        args.subfolder = "OmniMistral-v1_1/tokenizer"
    else:
        args.subfolder = None

    model = AutoModelForCausalLM.from_pretrained(args.orig_ckpt, subfolder="tuned-model", torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    # Загрузка аннотаций
    annotation_path = os.path.join(args.data_path, args.dataset)
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    print(f"Length of {annotation_path}: {len(data)}")

    args.save_file = os.path.join(args.exp_path, args.save_file)
    
    print(f"Monitoring experiment: {args.exp_path}")

    measures = []
    known_folders = []
    if os.path.exists(args.save_file):
        with open(args.save_file, 'r') as f:
            measures = json.load(f)
        for m in measures:
            known_folders.append(str(m["id"]))
    known_folders = set(known_folders)

    train_finished = False
    while True:
        current_folders = set(os.listdir(args.exp_path))
        if current_folders != known_folders:
            try:
                new_folders = current_folders - known_folders
                for folder in new_folders:
                    # check if Folder is folder
                    if not os.path.isdir(os.path.join(args.exp_path, folder)):
                        continue
                    print(f"New save detected: {folder}")
                    if folder == "tuned-model":
                        print("Detected 'tuned model' folder.")
                        train_finished = True
                        ckpt_path = args.exp_path
                        ckpt_id = -1
                    elif folder == "checkpoints":
                        continue
                    else:
                        ckpt_path = os.path.join(args.exp_path, folder)
                        ckpt_id = int(ckpt_path.split("/")[-1])

                    if str(ckpt_id) in known_folders:
                        continue

                    mean_metrics = evaluate_ckpt(args, data, ckpt_path, model)
                    print("Model: ", ckpt_path)
                    print("Mean IoU: ", mean_metrics["IoU"])
                    print("Mean Precision: ", mean_metrics["Precision"])
                    print("Mean Recall: ", mean_metrics["Recall"])
                    
                    measure = {
                        "id": ckpt_id,
                        "ckpt_path": ckpt_path,
                        "metrics": mean_metrics,
                    }
                    measures.append(measure)
                    with open(args.save_file, "w") as f:
                        f.write(json.dumps(measures, indent=4))

                known_folders = current_folders
            except Exception as ex:
                print("Exception", ex)

        if train_finished:
            break
        time.sleep(args.check_interval)
