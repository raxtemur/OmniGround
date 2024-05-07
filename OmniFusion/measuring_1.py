import torch
import os
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.request import urlopen
import torch.nn as nn
from huggingface_hub import hf_hub_download

from models import CLIPVisionTower
import re
import json
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import test_grounding_deamon as testing_daemon

DEVICE = "cuda:1"
PROMPT = "This is a dialog with AI assistant.\n"

    
def process_sample(sample):
    question = sample["conversations"][0]["value"]
    gt_ans = sample["conversations"][1]["value"]
    img_path = os.path.join("../data", os.path.join(sample["image"]))
    img = Image.open(img_path)

    answer = testing_daemon.gen_answer(
        model,
        tokenizer,
        clip,
        projection,
        query=question,
        special_embs=special_embs,
        image=img,
        device=DEVICE
    )

    return question, answer, gt_ans, img


if __name__ == "__main__":
    ckkpt_path = "../ckpts/Grouning_1B/version_19/2000"

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(os.path.join(ckkpt_path, "tuned-model"), torch_dtype=torch.bfloat16, device_map=DEVICE)

    projection = torch.load(os.path.join(ckkpt_path, "projection.pt"), map_location=DEVICE)
    special_embs = torch.load(os.path.join(ckkpt_path,"special_embeddings.pt"), map_location=DEVICE)

    clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    clip.load_model()
    clip = clip.to(device=DEVICE, dtype=torch.bfloat16)

    model.eval()
    clip.eval()
    print("Models loaded")

    test_paths = [
        "../data/OMNIGROUND_refcocog_umd_test_new.json",
        "../data/OMNIGROUND_refcocog_umd_val_new.json"
        # "../data/OMNIGROUND_refcoco+_unc_testB_new.json",
        # "../data/OMNIGROUND_refcoco+_unc_val_new.json",
    ]
    
    for path in test_paths:
        test_data = json.loads( open(path, "r").read())
        print("Loaded", path)
        
        acc05 = 0
        N = len(test_data)
        A = 0
        found = 0
        attempt = 0
        pxwise_metrics = []
        bar = tqdm(range(A, A+N), desc=f"Acc@05 = {0}:")
        for sample_id in bar:
            try:
                sample = test_data[sample_id]
                question, answer, gt_ans, img = process_sample(sample)                
                width, height = img.size
                
                bbox_gt = testing_daemon.get_bbox(gt_ans)
                bbox_gt_coorts = testing_daemon.rel_to_abs(bbox_gt, width, height)
                bbox_ans = testing_daemon.get_bbox(answer)
                
            except ValueError as ex:
                print(f"Incorrect output format. Sample id: {sample_id}; answer: {answer}; error: {ex} ")
                bbox_ans = [0, 0, 0, 0]
                found -= 1
            
            pxwise_metrics.append(testing_daemon.calculate_metrics(bbox_ans, bbox_gt))
            iou = pxwise_metrics[-1]["IoU"]
            if iou>0.5:
                acc05+=1
            found += 1
            bar.set_description(f"Acc@05 = {np.round(acc05/N, 3)}")
            bar.update(1)
        
        mean_pxwise_metrics = {k: np.sum([m[k] for m in pxwise_metrics])/N for k in pxwise_metrics[0].keys()}
        print("Accuracy@0.5:", acc05/found)
        print("Accuracy@0.5:", acc05/N)
        print("Mean pxwise metrics:", json.dumps(mean_pxwise_metrics, indent=4))
        print("\n\n")
        
        
        with open("./metrics_1.txt", "a") as f:
            f.write("Filename: " + path + "\n")
            f.write("Accuracy@0.5: " + str(acc05/found) + "\n")
            f.write("Accuracy@0.5: " + str(acc05/N) + "\n")
            f.write("Mean pxwise metrics: " + "\n")
            f.write(json.dumps(mean_pxwise_metrics, indent=4) + "\n\n")
        # end of loop over samples
        
        
