import ast
import gc
import os

import numpy as np
import pandas as pd
import torch
import json
from PIL import Image
from torch import nn
from tqdm import tqdm

import wandb
from utils import calculate_recall_at_K, encode_image, load_json


def gen_answer(cfg, model, tokenizer, clip, preprocess_val, projection, query, img_start_emb, img_end_emb, image=None):
    bad_words_ids = tokenizer(["\n", "</s>", ":", cfg.bot_name, "user", cfg.user_name], add_special_tokens=False).input_ids + [[13]]
    gen_params = {
        "do_sample": False,
        "max_new_tokens": 50,
        "early_stopping": True,
        "num_beams": 3,
        "repetition_penalty": 1.0,
        "remove_invalid_values": True,
        "eos_token_id": cfg.eos_id,
        "pad_token_id": cfg.eos_id,
        "forced_eos_token_id": cfg.eos_id,
        "use_cache": True,
        "no_repeat_ngram_size": 4,
        "bad_words_ids": bad_words_ids,
        "num_return_sequences": 1,
    }
    assert clip.visual.output_tokens
    with torch.no_grad():
        image_embedding = encode_image(image, clip, preprocess_val, cfg).to(device=model.device, dtype=model.dtype)
        projected_vision_embeddings = projection(image_embedding).to(device=model.device, dtype=model.dtype)
        prompt_ids = tokenizer.encode(f"{cfg.prompt}\n", add_special_tokens=False, return_tensors="pt").to(device=model.device)
        bs = projected_vision_embeddings.shape[0]
        ai_ids = tokenizer.encode(f"{cfg.user_name}{query}{cfg.bot_name}", add_special_tokens=False, return_tensors="pt").to(device=model.device)

        if cfg.training_params == "lora":
            prompt_embeddings = model.base_model.model.model.embed_tokens(prompt_ids)
            ai_embeddings = model.base_model.model.model.embed_tokens(ai_ids)
        elif cfg.training_params == "all":
            prompt_embeddings = model.model.embed_tokens(prompt_ids).to(model.dtype)
            ai_embeddings = model.model.embed_tokens(ai_ids).to(model.dtype)

        embeddings = torch.cat(
            [
                prompt_embeddings.repeat(bs, 1, 1),
                img_start_emb[None, None].repeat(bs, 1, 1),
                projected_vision_embeddings,
                img_end_emb[None, None].repeat(bs, 1, 1),
                ai_embeddings.repeat(bs, 1, 1),
            ],
            dim=1,
        ).to(dtype=model.dtype, device=model.device)

        out = model.generate(inputs_embeds=embeddings, **gen_params)
    out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts


def test(cfg, model, tokenizer, clip, preprocess_val, projection, img_start_emb, img_end_emb, log_prefix):
    root_datasets = "/home/jovyan/razzhigaev/DATASETS"
    test_images = {
        f"{root_datasets}/test_photos/digit0.png": "What is the number in the image?",
        f"{root_datasets}/test_photos/digit25.png": "What is the number in the image?",
        f"{root_datasets}/test_photos/time_10_30.png": "What time is it?",
        f"{root_datasets}/test_photos/elbrus.png": "What is its name?",
        f"{root_datasets}/test_photos/painting.png": "Who is the author?",
        f"{root_datasets}/test_photos/putin.jpeg": "Who is it?",
        f"{root_datasets}/test_photos/china.jpeg": "Who is it?",
        f"{root_datasets}/test_photos/fruits.jpeg": "What is it?",
        f"{root_datasets}/test_photos/cook.png": "What can I cook of this?",
        f"{root_datasets}/test_photos/titanic.png": "what is the ending of this movie?",
        f"{root_datasets}/test_photos/foot.jpg": "What is the diagnosis?",
        f"{root_datasets}/test_photos/meme.png": "Explain the meme in detail",
    }

    predicted_text = ""
    if wandb.run:
        answer_table = wandb.Table(columns=["image_id", "question", "answer"])
    for path in test_images.keys():
        query = test_images[path]
        answer = gen_answer(cfg, model, tokenizer, clip, preprocess_val, projection, query, img_start_emb, img_end_emb, Image.open(path))
        if wandb.run:
            image_id = path.split("/")[-1]
            answer_table.add_data(image_id, query, answer)
        predicted_text += f"\nQ: {query} \nA:{answer}"
    predicted_text = log_prefix + "\n" + predicted_text
    with open(f"ckpts/{cfg.exp_name}/test_dialogues.txt", "a", encoding="utf-8") as f:
        f.write(predicted_text)
    print("Updated test predicted text")

    if wandb.run:
        wandb.log({"test QA": answer_table})



def test_visual_dialog(cfg, model, tokenizer, clip, preprocess_val, projection, img_start_emb, img_end_emb, subsample=0.05):
    root_datasets = "/home/jovyan/kurkin/DATASETS"
    save_path = "visdial_ppl.csv"

    visdial = load_json(f"{root_datasets}/visdial/visdial_1.0_val.json")["data"]
    questions, answers, dialogs = visdial["questions"], visdial["answers"], visdial["dialogs"]
    with open('/home/jovyan/kurkin/DATASETS/visdial/visdial_1.0_val_short10ans_dialogs.json', 'r') as f:
        dialogs = json.load(f)
    if subsample:
        rng = np.random.default_rng(seed=132)
        dialogs = rng.choice(dialogs, size=int(subsample * len(dialogs)), replace=False)

    # if os.path.isfile(save_path):
    #     ppl_df = pd.read_csv(save_path)
    #     ppl_df["ppls"] = ppl_df["ppls"].apply(lambda x: ast.literal_eval(x))
    # else:
    #     ppl_df = pd.DataFrame(columns=["image_id", "question", "answer", "ppls"])
    #     ppl_df.to_csv(save_path, index=False)

    tokenizer.pad_token_id = cfg.pad_id

    modality_start_emb, modality_end_emb = img_start_emb, img_end_emb
    query = "Describe what you can see in the picture."
    # prompt_ids = tokenizer.encode(f"{cfg.prompt}\n{cfg.user_name}{query}", add_special_tokens=False, return_tensors="pt").to(device=model.device)
    prompt_ids = tokenizer.encode(f"{cfg.prompt}\n", add_special_tokens=False, return_tensors="pt").to(device=model.device) #
    prompt_embeddings = model.model.embed_tokens(prompt_ids).to(dtype=model.dtype)
    # ai_ids = tokenizer.encode(f"{cfg.bot_name}", add_special_tokens=False, return_tensors="pt").to(device=model.device)
    ai_ids = tokenizer.encode(f"{cfg.user_name}{query}{cfg.bot_name}", add_special_tokens=False, return_tensors="pt").to(device=model.device) #
    ai_embeddings = model.model.embed_tokens(ai_ids).to(dtype=model.dtype)

    loss = nn.CrossEntropyLoss(reduction="none", ignore_index=cfg.pad_id)
    dialogs_results = []
    for dialog in tqdm(dialogs):
        image_id = dialog["image_id"]
        image = f"{root_datasets}/visdial/VisualDialog_val2018/VisualDialog_val2018_{image_id:012d}.jpg"
        if not os.path.isfile(image):
            continue
        # 10 questions per image
        # if (ppl_df["image_id"] == image_id).sum() == 10:
        #     continue

        with torch.no_grad():
            img = Image.open(image).convert("RGB")
            img = encode_image(img, clip, preprocess_val, cfg)
            modality_embedding = img.to(
                device=model.device, dtype=next(projection.parameters()).dtype
            )
            projected_modality_embeddings = projection(modality_embedding).to(device=model.device, dtype=model.dtype)

        projection_embeddings = [
            modality_start_emb[None, None],
            projected_modality_embeddings.reshape(1, -1, cfg.emb_dim),
            modality_end_emb[None, None],
        ]
        caption_ids = tokenizer.encode(dialog["caption"], add_special_tokens=False, return_tensors="pt").to(device=model.device)
        caption_embeddings = model.model.embed_tokens(caption_ids)

        all_embeddings = torch.cat([prompt_embeddings, *projection_embeddings, ai_embeddings, caption_embeddings], dim=1).to(
            dtype=model.dtype, device=model.device
        )

        dialog_results = []
        for round_dialog in dialog["dialog"]:
            question_ids = tokenizer([str(cfg.user_name) + questions[round_dialog["question"]]], add_special_tokens=False, return_tensors="pt").to(
                device=model.device,
            )
            question_embeddings = model.model.embed_tokens(question_ids.input_ids)

            all_embeddings = torch.cat([all_embeddings, question_embeddings], dim=1).to(device=model.device, dtype=model.dtype)

            options = [answers[opt] for opt in round_dialog["answer_options"]]
            answer_ids = tokenizer(options, add_special_tokens=False, padding=True, return_tensors="pt").to(device=model.device)
            answer_option_embeddings = model.model.embed_tokens(answer_ids.input_ids)

            dialogue_embeddings = torch.cat([all_embeddings, ai_embeddings], dim=1).repeat(answer_option_embeddings.shape[0], 1, 1)
            dialogue_embeddings = torch.cat([dialogue_embeddings, answer_option_embeddings], dim=1).to(device=model.device, dtype=model.dtype)
            attention_mask_answer = answer_ids.attention_mask
            attention_mask = torch.ones(dialogue_embeddings.shape[:2])
            attention_mask[:, -attention_mask_answer.shape[1] :] = attention_mask_answer
            attention_mask = attention_mask.to(device=model.device)

            with torch.no_grad():
                out_logits = model(inputs_embeds=dialogue_embeddings, attention_mask=attention_mask).logits
            shift_logits = out_logits[:, -answer_option_embeddings.shape[1] - 1 : -1].contiguous()
            labels = answer_ids.input_ids

            neg_log_likelihood = (loss(shift_logits.transpose(1, 2), labels) * attention_mask_answer).sum(1) / attention_mask_answer.sum(1)
            ppl_result = {
                "image_id": image_id,
                "question": round_dialog["question"],
                "answer": round_dialog["answer"],
                "ppls": dict(zip(round_dialog["answer_options"], neg_log_likelihood.tolist())),
            }
            dialog_results.append(ppl_result)

            true_answer_ids = tokenizer([answers[round_dialog["answer"]]], add_special_tokens=False, return_tensors="pt").to(
                device=model.device,
            )
            true_answer_embeddings = model.model.embed_tokens(true_answer_ids.input_ids)
            all_embeddings = torch.cat([all_embeddings, ai_embeddings, true_answer_embeddings], dim=1).to(device=model.device, dtype=model.dtype)
            del dialogue_embeddings, answer_option_embeddings, out_logits, shift_logits, attention_mask_answer, attention_mask
            gc.collect()
            torch.cuda.empty_cache()
        # new_ppl_df = pd.DataFrame.from_records(dialog_results)
        # ppl_df = pd.concat([ppl_df, new_ppl_df])
        # new_ppl_df.to_csv("visdial_ppl.csv", mode="a", header=False, index=False)
        del all_embeddings
        gc.collect()
        torch.cuda.empty_cache()
        dialogs_results.append(dialog_results)

    len_ = 0
    correct = 0
    for i in range(len(dialogs_results)):
        for j in range(len(dialogs_results[i])):
            scores = dialogs_results[i][j]['ppls']
            
            prediction = list(scores.keys())[0]
            for ans_id in scores.keys():
                if scores[ans_id] < scores[prediction]:
                    prediction = ans_id
            if dialogs_results[i][j]['answer'] == prediction:
                correct += 1
            len_ += 1
    acc = correct / len_
    
    return acc
