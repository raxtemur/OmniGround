import argparse
import json
import re
import os

from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_src.dataset import get_collate_function, get_dataset
from models import CLIPVisionTower
import warnings
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


def gen_answer(model, tokenizer, clip, projection, query, special_embs, image=None):
    gen_params = get_gen_params(tokenizer)
    
    with torch.no_grad():
        image_features = clip.image_processor(image, return_tensors='pt')
        image_embedding = clip(image_features['pixel_values']).to(device=DEVICE, dtype=torch.bfloat16)

        projected_vision_embeddings = projection(image_embedding).to(device=DEVICE, dtype=torch.bfloat16)
        prompt_ids = tokenizer.encode(f"{PROMPT}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
        question_ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)

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
        ).to(dtype=torch.bfloat16, device=DEVICE)
        out = model.generate(inputs_embeds=embeddings, **gen_params)
    out = out[:, :-1]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts

"""
def gen_answers_on_dataset(model, tokenizer, clip, projection, cfg, special_embs):
    gen_params = get_gen_params(tokenizer)
    
    test_dataset = get_dataset(cfg, tokenizer, clip.image_processor)
    collate_function = get_collate_function(cfg)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=collate_function)
    
    generated_texts = []
    
    for i, batch in enumerate(tqdm(test_dataloader)):
        images, images_mask, labels, mask, positions = batch
        if images_mask.sum() > 0:
            image_embedding = clip(images).to(dtype=torch.bfloat16) # preprocessing!!!
            projected_vision_embeddings = projection(image_embedding)
        
        embeddings = model.model.embed_tokens(labels)
        img_idx_counter = 0
        for i in range(len(embeddings)):
            for pos in positions[i]:
                
                if pos['type'] in special_embs.keys():
                    embeddings[i][pos['position']] = special_embs[pos['type']]
                if pos['type'] == 'IMG':
                    embeddings[i][pos['position'][0]:pos['position'][1]] = projected_vision_embeddings[img_idx_counter]
                    img_idx_counter += 1
        
        embeddings = embeddings[:, :cfg.max_context_len]
        labels = labels[:, :cfg.max_context_len]
        # mask = mask[:, :cfg.max_context_len]
        
        generated_texts.extend(model.generate(inputs_embeds=embeddings.to(dtype = torch.bfloat16), **gen_params))
    return generated_texts
""" 

def correct_bbox(bbox):
    left, top, right, bottom = bbox
    
    new_bottom = min(max(top, bottom), 1)
    new_top = min(top, bottom)
    
    new_left = min(left, right)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--ckpt_path', type=str, default="./ckpts/Grouning_0/version_2/49999")
    parser.add_argument('--data_path', type=str, default="./data/")
    parser.add_argument('--dataset', type=str, default="CWB_flickr30k_test_short_ref.json")
    parser.add_argument('--tokenizer', type=str, default="AIRI-Institute/OmniFusion")
    parser.add_argument('--save_file', type=str, default="./metrics.txt")
    args = parser.parse_args()

    if args.tokenizer == "AIRI-Institute/OmniFusion":
        args.subfolder = "OmniMistral-v1_1/tokenizer"
    else:
        args.subfolder = None

    # Загрузка аннотаций
    annotation_path = os.path.join(args.data_path, args.dataset)
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    print(f"Length of {annotation_path}: {len(data)}")

    # Загрузка модели
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, subfolder=args.subfolder, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, subfolder="tuned-model", torch_dtype=torch.bfloat16, device_map=DEVICE)
    # model = AutoModelForCausalLM.from_pretrained(os.path.join(args.ckpt_path, "tuned-model"), torch_dtype=torch.bfloat16, device_map=DEVICE)
    projection = torch.load(os.path.join(args.ckpt_path, "projection.pt"), map_location=DEVICE)
    special_embs = torch.load(os.path.join(args.ckpt_path,"special_embeddings.pt"), map_location=DEVICE)
    clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    clip.load_model()
    clip = clip.to(device=DEVICE, dtype=torch.bfloat16)
    torch.inference_mode()
    model.eval()
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
                query=question,
                special_embs=special_embs,
                image=img
            )

            bbox_ans = get_bbox(answer)
            bbox_gt = get_bbox(gt_ans)

            # Получение размеров изображения
            width, height = img.size
            bbox_ans_coors = rel_to_abs(bbox_ans, width, height)
            bbox_gt_coorts = rel_to_abs(bbox_gt, width, height)

            metrics.append(calculate_metrics(bbox_ans, bbox_gt))
        except ValueError as ex:
            warnings.warn(f"Incorrect output format. Sample id: {sample_id}; answer: {answer}; error: {ex} ")
            metrics.append(calculate_metrics(bbox_ans, [0, 0, 0, 0]))
    
    mean_metrics = {k: np.sum([m[k] for m in metrics])/args.num_samples for k in metrics[0].keys()}
    
    print("Model: ", args.ckpt_path)
    print("Mean IoU: ", mean_metrics["IoU"])
    print("Mean Accuracy: ", mean_metrics["Accuracy"])
    
    data = {
        "id": args.ckpt_path.split("/")[-1],
        "ckpt_path": args.ckpt_path,
        "metrics": mean_metrics,
    }
    # дописать в файл
    with open(args.save_file, "a") as f:
        f.write(json.dumps(data, indent=4))
