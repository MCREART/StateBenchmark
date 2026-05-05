#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import math
import os
import random
import re
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models.yaml"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "next_state_vector"
EMB_DIR = OUT_DIR / "embeddings"
RUN_DIR = OUT_DIR / "runs"
ERROR_LOG = OUT_DIR / "errors.log"

DEFAULT_MODELS = ["Qwen3-Embedding-0.6B", "bge-large-en-v1.5", "e5-large-v2"]
SPLIT_COUNTS = {"train": 6000, "dev": 800, "test": 1200}

COLORS = ["红色", "蓝色", "绿色", "黄色", "紫色", "黑色", "白色", "银色"]
SIZES = ["很小", "小", "中等大小", "大", "很大"]
MATERIALS = ["木质", "金属", "塑料", "玻璃", "陶瓷", "纸质"]
TEMPERATURES = ["冰冷", "冷", "常温", "温热", "很热"]
SWITCH_STATES = ["关着", "开着"]
WETNESS = ["湿的", "干的"]
OBJECTS = [
    "球", "盒子", "杯子", "书", "钥匙", "灯", "衣服", "桌子", "瓶子", "玩具车", "盘子", "猫玩偶",
    "遥控器", "手机", "笔记本", "硬币", "刷子", "积木", "闹钟", "马克笔", "花瓶", "毛巾", "纸巾",
    "水杯", "勺子", "耳机", "钱包", "相机", "手套", "帽子", "枕头", "玩具熊", "尺子", "剪刀",
]
PLACES = [
    "桌子", "椅子", "书架", "地板", "窗台", "柜台", "床", "垫子",
    "沙发", "门口", "厨房台面", "茶几", "工作台", "阳台", "墙角", "楼梯",
]
CONTAINERS = [
    "书包", "盒子", "抽屉", "篮子", "柜子", "袋子",
    "箱子", "罐子", "杯柜", "收纳盒", "行李箱", "纸袋",
]
ROOMS = ["房间", "厨房", "卧室", "客厅", "办公室", "教室", "走廊", "储物间"]
DIRECTIONS = [
    ("左边", "右边", "移到"),
    ("前面", "后面", "移到"),
    ("上方", "下方", "移到"),
    ("里面", "外面", "移到"),
]


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def normalize_next_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def next_text_group_ids(texts):
    group_by_text = {}
    ids = []
    for text in texts:
        key = normalize_next_text(text)
        if key not in group_by_text:
            group_by_text[key] = len(group_by_text)
        ids.append(group_by_text[key])
    return np.asarray(ids, dtype=np.int64)


def log_error(label, exc):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {label}\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        f.write("\n")


def other(rng, values, current):
    return rng.choice([v for v in values if v != current])


def make_attr_case(rng, split, idx):
    kind = rng.choice(["color", "size", "material", "temperature", "switch", "wetness"])
    obj = rng.choice(OBJECTS)
    place = rng.choice(PLACES)
    if kind == "color":
        old, new = rng.choice(COLORS), None
        new = other(rng, COLORS, old)
        text_t = rng.choice([f"{place}上有一个{old}的{obj}", f"{old}{obj}在{place}上", f"一个{old}的{obj}放在{place}上"])
        action = rng.choice([f"把{obj}刷成{new}", f"给{obj}换成{new}", f"把{obj}涂成{new}"])
        text_t1 = rng.choice([f"{place}上有一个{new}的{obj}", f"{new}{obj}在{place}上", f"一个{new}的{obj}放在{place}上"])
    elif kind == "size":
        old, new = rng.choice(SIZES), None
        new = other(rng, SIZES, old)
        action_word = "放大" if SIZES.index(new) > SIZES.index(old) else "缩小"
        text_t = rng.choice([f"{place}上有一个{old}的{obj}", f"{old}{obj}在{place}上"])
        action = rng.choice([f"把{obj}{action_word}", f"将{obj}变成{new}", f"调整{obj}大小到{new}"])
        text_t1 = rng.choice([f"{place}上有一个{new}的{obj}", f"{new}{obj}在{place}上"])
    elif kind == "material":
        old, new = rng.choice(MATERIALS), None
        new = other(rng, MATERIALS, old)
        text_t = rng.choice([f"{old}{obj}在{place}上", f"{place}上放着一个{old}{obj}"])
        action = rng.choice([f"把{obj}换成{new}材质", f"将{obj}改造成{new}", f"给{obj}换成{new}外壳"])
        text_t1 = rng.choice([f"{new}{obj}在{place}上", f"{place}上放着一个{new}{obj}"])
    elif kind == "temperature":
        old, new = rng.choice(TEMPERATURES), None
        new = other(rng, TEMPERATURES, old)
        verb = "加热" if TEMPERATURES.index(new) > TEMPERATURES.index(old) else "冷却"
        text_t = rng.choice([f"{place}上有一杯{old}的水", f"{old}的水在{place}上"])
        action = rng.choice([f"把水{verb}", f"让水变成{new}", f"处理这杯水直到它{new}"])
        text_t1 = rng.choice([f"{place}上有一杯{new}的水", f"{new}的水在{place}上"])
        obj = "水"
    elif kind == "switch":
        old = rng.choice(SWITCH_STATES)
        new = other(rng, SWITCH_STATES, old)
        verb = "打开" if new == "开着" else "关掉"
        obj = rng.choice(["灯", "风扇", "开关", "台灯"])
        text_t = rng.choice([f"{obj}{old}", f"{place}上的{obj}{old}"])
        action = rng.choice([f"{verb}{obj}", f"把{obj}设为{new}", f"切换{obj}状态"])
        text_t1 = rng.choice([f"{obj}{new}", f"{place}上的{obj}{new}"])
    else:
        old = rng.choice(WETNESS)
        new = other(rng, WETNESS, old)
        verb = "晒干" if new == "干的" else "弄湿"
        obj = rng.choice(["衣服", "毛巾", "纸巾", "布"])
        text_t = rng.choice([f"{obj}是{old}", f"{place}上的{obj}是{old}"])
        action = rng.choice([f"把{obj}{verb}", f"让{obj}变成{new}", f"处理{obj}直到它是{new}"])
        text_t1 = rng.choice([f"{obj}是{new}", f"{place}上的{obj}是{new}"])
    return {
        "id": f"{split}_attr_{idx:06d}",
        "split": split,
        "range": "attribute",
        "transition_type": kind,
        "text_t": text_t,
        "action_text": action,
        "text_t1": text_t1,
    }


def make_relation_case(rng, split, idx):
    kind = rng.choice(["container_in", "container_out", "left_right", "right_left", "under_on", "on_under", "outside_inside", "inside_outside"])
    obj = rng.choice([o for o in OBJECTS if o not in ["桌子"]])
    container = rng.choice(CONTAINERS)
    place = rng.choice(PLACES)
    if kind == "container_in":
        text_t = rng.choice([f"{obj}在{container}外面", f"{obj}不在{container}里", f"{obj}放在{container}旁边"])
        action = rng.choice([f"把{obj}放进{container}", f"将{obj}装进{container}", f"把{obj}收进{container}"])
        text_t1 = rng.choice([f"{obj}在{container}里", f"{container}里有{obj}", f"{obj}被放进{container}"])
    elif kind == "container_out":
        text_t = rng.choice([f"{obj}在{container}里", f"{container}里有{obj}"])
        action = rng.choice([f"拿出{obj}", f"把{obj}从{container}里取出来", f"将{obj}移到{container}外"])
        text_t1 = rng.choice([f"{obj}在{container}外面", f"{obj}不在{container}里", f"{obj}被拿到{container}外"])
    elif kind == "left_right":
        text_t = f"{obj}在{place}左边"
        action = f"把{obj}移到{place}右边"
        text_t1 = f"{obj}在{place}右边"
    elif kind == "right_left":
        text_t = f"{obj}在{place}右边"
        action = f"把{obj}移到{place}左边"
        text_t1 = f"{obj}在{place}左边"
    elif kind == "under_on":
        text_t = f"{obj}在{place}下面"
        action = f"把{obj}放到{place}上"
        text_t1 = f"{obj}在{place}上"
    elif kind == "on_under":
        text_t = f"{obj}在{place}上"
        action = f"把{obj}放到{place}下面"
        text_t1 = f"{obj}在{place}下面"
    elif kind == "outside_inside":
        text_t = f"{obj}在房间外"
        action = f"把{obj}搬进房间"
        text_t1 = f"{obj}在房间里"
    else:
        text_t = f"{obj}在房间里"
        action = f"把{obj}搬到房间外"
        text_t1 = f"{obj}在房间外"
    return {
        "id": f"{split}_rel_{idx:06d}",
        "split": split,
        "range": "relation",
        "transition_type": kind,
        "text_t": text_t,
        "action_text": action,
        "text_t1": text_t1,
    }


def dedupe_candidates(candidates):
    seen = {}
    for row in candidates:
        key = (row["text_t"], row["action_text"], row["text_t1"])
        seen.setdefault(key, row)
    return list(seen.values())


def enumerate_attribute_candidates():
    candidates = []
    for obj in OBJECTS:
        for place in PLACES:
            for old in COLORS:
                for new in COLORS:
                    if old == new:
                        continue
                    candidates.append({
                        "range": "attribute",
                        "transition_type": "color",
                        "text_t": f"{place}上有一个{old}的{obj}",
                        "action_text": f"把{obj}刷成{new}",
                        "text_t1": f"{place}上有一个{new}的{obj}",
                    })
                    candidates.append({
                        "range": "attribute",
                        "transition_type": "color",
                        "text_t": f"{old}{obj}在{place}上",
                        "action_text": f"把{obj}涂成{new}",
                        "text_t1": f"{new}{obj}在{place}上",
                    })
            for old in SIZES:
                for new in SIZES:
                    if old == new:
                        continue
                    verb = "放大" if SIZES.index(new) > SIZES.index(old) else "缩小"
                    candidates.append({
                        "range": "attribute",
                        "transition_type": "size",
                        "text_t": f"{place}上有一个{old}的{obj}",
                        "action_text": f"把{obj}{verb}到{new}",
                        "text_t1": f"{place}上有一个{new}的{obj}",
                    })
            for old in MATERIALS:
                for new in MATERIALS:
                    if old == new:
                        continue
                    candidates.append({
                        "range": "attribute",
                        "transition_type": "material",
                        "text_t": f"{old}{obj}在{place}上",
                        "action_text": f"把{obj}换成{new}材质",
                        "text_t1": f"{new}{obj}在{place}上",
                    })
    for place in PLACES:
        for old in TEMPERATURES:
            for new in TEMPERATURES:
                if old == new:
                    continue
                verb = "加热" if TEMPERATURES.index(new) > TEMPERATURES.index(old) else "冷却"
                candidates.append({
                    "range": "attribute",
                    "transition_type": "temperature",
                    "text_t": f"{place}上有一杯{old}的水",
                    "action_text": f"把水{verb}到{new}",
                    "text_t1": f"{place}上有一杯{new}的水",
                })
    for obj in ["灯", "风扇", "开关", "台灯", "手电筒", "电视"]:
        for place in PLACES:
            for old in SWITCH_STATES:
                new = "开着" if old == "关着" else "关着"
                verb = "打开" if new == "开着" else "关掉"
                candidates.append({
                    "range": "attribute",
                    "transition_type": "switch",
                    "text_t": f"{place}上的{obj}{old}",
                    "action_text": f"{verb}{obj}",
                    "text_t1": f"{place}上的{obj}{new}",
                })
    for obj in ["衣服", "毛巾", "纸巾", "布", "手套", "帽子"]:
        for place in PLACES:
            for old in WETNESS:
                new = "干的" if old == "湿的" else "湿的"
                verb = "晒干" if new == "干的" else "弄湿"
                candidates.append({
                    "range": "attribute",
                    "transition_type": "wetness",
                    "text_t": f"{place}上的{obj}是{old}",
                    "action_text": f"把{obj}{verb}",
                    "text_t1": f"{place}上的{obj}是{new}",
                })
    return dedupe_candidates(candidates)


def enumerate_relation_candidates():
    candidates = []
    relation_objects = [o for o in OBJECTS if o not in {"桌子"}]
    for obj in relation_objects:
        for container in CONTAINERS:
            for text_t in [f"{obj}在{container}外面", f"{obj}不在{container}里", f"{obj}放在{container}旁边", f"{obj}靠在{container}外侧"]:
                for action in [f"把{obj}放进{container}", f"将{obj}装进{container}", f"把{obj}收进{container}", f"把{obj}移入{container}"]:
                    for text_t1 in [f"{obj}在{container}里", f"{container}里有{obj}", f"{obj}被放进{container}", f"{obj}位于{container}内部"]:
                        candidates.append({
                            "range": "relation",
                            "transition_type": "container_in",
                            "text_t": text_t,
                            "action_text": action,
                            "text_t1": text_t1,
                        })
            for text_t in [f"{obj}在{container}里", f"{container}里有{obj}", f"{obj}位于{container}内部"]:
                for action in [f"拿出{obj}", f"把{obj}从{container}里取出来", f"将{obj}移到{container}外", f"把{obj}取出{container}"]:
                    for text_t1 in [f"{obj}在{container}外面", f"{obj}不在{container}里", f"{obj}被拿到{container}外", f"{obj}放在{container}旁边"]:
                        candidates.append({
                            "range": "relation",
                            "transition_type": "container_out",
                            "text_t": text_t,
                            "action_text": action,
                            "text_t1": text_t1,
                        })
        for place in PLACES:
            for left, right, verb in DIRECTIONS:
                candidates.append({
                    "range": "relation",
                    "transition_type": f"{left}_to_{right}",
                    "text_t": f"{obj}在{place}{left}",
                    "action_text": f"把{obj}{verb}{place}{right}",
                    "text_t1": f"{obj}在{place}{right}",
                })
                candidates.append({
                    "range": "relation",
                    "transition_type": f"{right}_to_{left}",
                    "text_t": f"{obj}在{place}{right}",
                    "action_text": f"把{obj}{verb}{place}{left}",
                    "text_t1": f"{obj}在{place}{left}",
                })
            candidates.append({
                "range": "relation",
                "transition_type": "under_on",
                "text_t": f"{obj}在{place}下面",
                "action_text": f"把{obj}放到{place}上",
                "text_t1": f"{obj}在{place}上",
            })
            candidates.append({
                "range": "relation",
                "transition_type": "on_under",
                "text_t": f"{obj}在{place}上",
                "action_text": f"把{obj}放到{place}下面",
                "text_t1": f"{obj}在{place}下面",
            })
        for room in ROOMS:
            candidates.append({
                "range": "relation",
                "transition_type": "outside_inside",
                "text_t": f"{obj}在{room}外",
                "action_text": f"把{obj}搬进{room}",
                "text_t1": f"{obj}在{room}里",
            })
            candidates.append({
                "range": "relation",
                "transition_type": "inside_outside",
                "text_t": f"{obj}在{room}里",
                "action_text": f"把{obj}搬到{room}外",
                "text_t1": f"{obj}在{room}外",
            })
    return dedupe_candidates(candidates)


def sample_without_replacement(rng, candidates, total, label):
    if len(candidates) < total:
        raise ValueError(f"not enough unique {label} candidates: need {total}, have {len(candidates)}")
    candidates = list(candidates)
    rng.shuffle(candidates)
    return candidates[:total]


def generate_data(seed, overwrite=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    total_attr = sum((count + 1) // 2 for count in SPLIT_COUNTS.values())
    total_rel = sum(count // 2 for count in SPLIT_COUNTS.values())
    attr_pool = enumerate_attribute_candidates()
    rel_pool = enumerate_relation_candidates()
    attr_samples = sample_without_replacement(rng, attr_pool, total_attr, "attribute")
    rel_samples = sample_without_replacement(rng, rel_pool, total_rel, "relation")
    print(f"candidate_pool attribute={len(attr_pool)} relation={len(rel_pool)}")
    print(f"sampled_without_replacement attribute={len(attr_samples)} relation={len(rel_samples)}")
    attr_offset = 0
    rel_offset = 0
    for split, count in SPLIT_COUNTS.items():
        path = DATA_DIR / f"next_state_vector_{split}.jsonl"
        if path.exists() and not overwrite:
            print(f"data exists: {path}")
            attr_offset += (count + 1) // 2
            rel_offset += count // 2
            continue
        split_attr = attr_samples[attr_offset:attr_offset + (count + 1) // 2]
        split_rel = rel_samples[rel_offset:rel_offset + count // 2]
        attr_offset += (count + 1) // 2
        rel_offset += count // 2
        rows = []
        for i in range(count):
            source = split_attr[i // 2] if i % 2 == 0 else split_rel[i // 2]
            row = dict(source)
            row["id"] = f"{split}_{row['range'][:4]}_{i:06d}"
            row["split"] = split
            rows.append(row)
        keys = [(r["text_t"], r["action_text"], r["text_t1"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise AssertionError(f"duplicate samples in split {split}")
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({count} rows)")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_models(selected, models_yaml):
    with open(models_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    by_name = {m["name"]: m for m in data.get("models", [])}
    models = []
    for name in selected:
        if name not in by_name:
            raise KeyError(f"model not found in {models_yaml}: {name}")
        models.append(by_name[name])
    return models


def load_encoder(model_cfg):
    kwargs = {}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    return SentenceTransformer(
        model_cfg["path"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        local_files_only=True,
        model_kwargs=kwargs,
    )


def encode_with_fallback(model, texts, batch_size, prefix, label):
    prepared = [prefix + t for t in texts] if prefix else list(texts)
    bs = int(batch_size)
    while bs >= 4:
        try:
            emb = model.encode(
                prepared,
                batch_size=bs,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            return np.asarray(emb, dtype=np.float32), bs
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg and "cuda" not in msg:
                raise
            print(f"OOM in {label} at batch_size={bs}; retrying smaller", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            bs //= 2
    raise RuntimeError(f"could not encode {label}; batch size fell below 4")


def cache_embeddings(model_cfg, splits, overwrite=False, default_batch_size=64):
    name = model_cfg["name"]
    prefix = model_cfg.get("prefix", "") or ""
    batch_size = int(model_cfg.get("batch_size") or default_batch_size)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    missing = [s for s in splits if overwrite or not (EMB_DIR / f"{safe_name(name)}_{s}.npz").exists()]
    if not missing:
        print(f"{name}: caches exist; skipping")
        return
    print(f"\n===== CACHE {name} {missing} =====", flush=True)
    encoder = load_encoder(model_cfg)
    try:
        for split in missing:
            rows = read_jsonl(DATA_DIR / f"{DATA_PREFIX}_{split}.jsonl")
            z_t, bs1 = encode_with_fallback(encoder, [r["text_t"] for r in rows], batch_size, prefix, f"{name}:{split}:state")
            z_a, bs2 = encode_with_fallback(encoder, [r["action_text"] for r in rows], batch_size, prefix, f"{name}:{split}:action")
            z_next, bs3 = encode_with_fallback(encoder, [r["text_t1"] for r in rows], batch_size, prefix, f"{name}:{split}:next")
            out = EMB_DIR / f"{safe_name(name)}_{split}.npz"
            np.savez_compressed(
                out,
                z_t=z_t,
                z_action=z_a,
                z_next=z_next,
                ids=np.asarray([r["id"] for r in rows], dtype=str),
                ranges=np.asarray([r["range"] for r in rows], dtype=str),
                transition_types=np.asarray([r["transition_type"] for r in rows], dtype=str),
                text_t=np.asarray([r["text_t"] for r in rows], dtype=str),
                action_text=np.asarray([r["action_text"] for r in rows], dtype=str),
                text_t1=np.asarray([r["text_t1"] for r in rows], dtype=str),
            )
            print(f"wrote {out} z={z_t.shape} batch_sizes=({bs1},{bs2},{bs3})", flush=True)
    finally:
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class NextStateVectorConcatMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z_t, z_action):
        return self.net(torch.cat([z_t, z_action], dim=1))


class NextStateVectorSingleInputMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, source):
        super().__init__()
        self.source = source
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z_t, z_action):
        x = z_t if self.source == "state_only" else z_action
        return self.net(x)


class NextStateVectorFiLM(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, residual=True):
        super().__init__()
        self.residual = residual
        self.action_to_film = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim * 2),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z_t, z_action):
        gamma_beta = self.action_to_film(z_action)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        modulated = gamma * z_t + beta
        if self.residual:
            modulated = modulated + z_t
        return self.output(modulated)


class NextStateVectorGatedResidualFiLM(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.action_to_params = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim * 3),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z_t, z_action):
        params = self.action_to_params(z_action)
        gamma, beta, gate_logits = params.chunk(3, dim=1)
        gate = torch.sigmoid(gate_logits)
        delta = gamma * z_t + beta
        modulated = z_t + gate * delta
        return self.output(modulated)


class NextStateVectorPairGatedResidual(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        pair_dim = dim * 3
        self.gate = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.delta = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z_t, z_action):
        pair = torch.cat([z_t, z_action, z_t * z_action], dim=1)
        gate = torch.sigmoid(self.gate(pair))
        delta = self.delta(pair)
        modulated = z_t + gate * delta
        return self.output(modulated)


def make_loader(data, batch_size, shuffle, include_next_text_groups=False):
    x1 = torch.from_numpy(np.asarray(data["z_t"], dtype=np.float32))
    x2 = torch.from_numpy(np.asarray(data["z_action"], dtype=np.float32))
    y = torch.from_numpy(np.asarray(data["z_next"], dtype=np.float32))
    if include_next_text_groups:
        groups = torch.from_numpy(next_text_group_ids(data["text_t1"]))
        return DataLoader(TensorDataset(x1, x2, y, groups), batch_size=batch_size, shuffle=shuffle)
    return DataLoader(TensorDataset(x1, x2, y), batch_size=batch_size, shuffle=shuffle)


def predict_vectors(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            z_t, z_action, y = batch[:3]
            pred = model(z_t.to(device), z_action.to(device))
            pred = F.normalize(pred, dim=1).cpu().numpy()
            preds.append(pred)
            trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def masked_infonce_loss(logits, labels, next_text_groups, policy):
    if policy == "none":
        return F.cross_entropy(logits, labels)
    if policy != "same_next_text":
        raise ValueError(f"unknown false-negative mask policy: {policy}")
    same_next = next_text_groups[:, None].eq(next_text_groups[None, :])
    diagonal = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    false_negative_mask = same_next & ~diagonal
    masked_logits = logits.masked_fill(false_negative_mask, torch.finfo(logits.dtype).min)
    return F.cross_entropy(masked_logits, labels)


def retrieval_metrics(pred, true, topks=(1, 5, 10)):
    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    true = true / np.maximum(np.linalg.norm(true, axis=1, keepdims=True), 1e-8)
    sims = pred @ true.T
    ranks = np.argsort(-sims, axis=1)
    out = {}
    target = np.arange(true.shape[0])
    for k in topks:
        out[f"retrieval_top{k}"] = float(np.mean([target[i] in ranks[i, :k] for i in range(true.shape[0])]))
    out["mean_target_rank"] = float(np.mean([int(np.where(ranks[i] == i)[0][0]) + 1 for i in range(true.shape[0])]))
    return out, sims


def evaluate(pred, true):
    pred_n = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    true_n = true / np.maximum(np.linalg.norm(true, axis=1, keepdims=True), 1e-8)
    cos = np.sum(pred_n * true_n, axis=1)
    mse = np.mean((pred_n - true_n) ** 2, axis=1)
    ret, _ = retrieval_metrics(pred_n, true_n)
    out = {
        "cosine_mean": float(cos.mean()),
        "cosine_median": float(np.median(cos)),
        "cosine_p10": float(np.percentile(cos, 10)),
        "mse_mean": float(mse.mean()),
    }
    out.update(ret)
    return out, cos


def split_metrics(pred, true, labels):
    metrics = {}
    labels = np.asarray(labels).astype(str)
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if int(mask.sum()) == 0:
            continue
        m, _ = evaluate(pred[mask], true[mask])
        m["n"] = int(mask.sum())
        metrics[label] = m
    return metrics


def train_one(model_cfg, args):
    name = model_cfg["name"]
    train = np.load(EMB_DIR / f"{safe_name(name)}_train.npz", allow_pickle=False)
    dev = np.load(EMB_DIR / f"{safe_name(name)}_dev.npz", allow_pickle=False)
    test = np.load(EMB_DIR / f"{safe_name(name)}_test.npz", allow_pickle=False)
    dim = int(train["z_next"].shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.architecture == "concat":
        model = NextStateVectorConcatMLP(dim, args.hidden_dim, args.dropout).to(device)
    elif args.architecture in {"state_only", "action_only"}:
        model = NextStateVectorSingleInputMLP(dim, args.hidden_dim, args.dropout, args.architecture).to(device)
    elif args.architecture in {"film", "residual_film"}:
        model = NextStateVectorFiLM(dim, args.hidden_dim, args.dropout, residual=args.architecture == "residual_film").to(device)
    elif args.architecture == "gated_residual_film":
        model = NextStateVectorGatedResidualFiLM(dim, args.hidden_dim, args.dropout).to(device)
    elif args.architecture == "pair_gated_residual":
        model = NextStateVectorPairGatedResidual(dim, args.hidden_dim, args.dropout).to(device)
    else:
        raise ValueError(args.architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(train, args.train_batch_size, True, include_next_text_groups=args.false_negative_mask != "none")
    dev_loader = make_loader(dev, args.eval_batch_size, False)
    test_loader = make_loader(test, args.eval_batch_size, False)

    best_state, best_epoch, best_dev_cos, stale = None, -1, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            z_t, z_action, y = batch[:3]
            z_t = z_t.to(device)
            z_action = z_action.to(device)
            y = y.to(device)
            next_text_groups = batch[3].to(device) if len(batch) > 3 else None
            pred = model(z_t, z_action)
            pred_n = F.normalize(pred, dim=1)
            y_n = F.normalize(y, dim=1)
            logits = pred_n @ y_n.T / args.temperature
            labels = torch.arange(logits.size(0), device=device)
            if next_text_groups is None:
                loss = F.cross_entropy(logits, labels)
            else:
                loss = masked_infonce_loss(logits, labels, next_text_groups, args.false_negative_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * z_t.size(0)
            seen += z_t.size(0)
        dev_pred, dev_true = predict_vectors(model, dev_loader, device)
        dev_metrics, _ = evaluate(dev_pred, dev_true)
        dev_cos = dev_metrics["cosine_mean"]
        print(f"{name} epoch={epoch} loss={total_loss/seen:.6f} dev_cos={dev_cos:.4f} dev_top1={dev_metrics['retrieval_top1']:.4f}", flush=True)
        if dev_cos > best_dev_cos:
            best_dev_cos = dev_cos
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    dev_pred, dev_true = predict_vectors(model, dev_loader, device)
    test_pred, test_true = predict_vectors(model, test_loader, device)
    dev_metrics, _ = evaluate(dev_pred, dev_true)
    test_metrics, test_cos = evaluate(test_pred, test_true)

    by_range = split_metrics(test_pred, test_true, test["ranges"])
    by_transition = split_metrics(test_pred, test_true, test["transition_types"])

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    model_file = RUN_DIR / f"{safe_name(name)}_next_state_vector.pt"
    torch.save({
        "model": name,
        "state_dict": model.state_dict(),
        "dim": dim,
        "hidden_dim": args.hidden_dim,
        "architecture": args.architecture,
        "false_negative_mask": args.false_negative_mask,
        "best_epoch": best_epoch,
    }, model_file)

    metrics = {
        "model": name,
        "architecture": args.architecture,
        "false_negative_mask": args.false_negative_mask,
        "embedding_dim": dim,
        "num_train": int(train["z_next"].shape[0]),
        "num_dev": int(dev["z_next"].shape[0]),
        "num_test": int(test["z_next"].shape[0]),
        "best_epoch": int(best_epoch),
        "dev": dev_metrics,
        "test": test_metrics,
        "test_by_range": by_range,
        "test_by_transition_type": by_transition,
    }
    metrics_file = RUN_DIR / f"{safe_name(name)}_next_state_vector_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    pred_file = RUN_DIR / f"{safe_name(name)}_next_state_vector_predictions_test.csv"
    nn = test_pred @ (test_true / np.maximum(np.linalg.norm(test_true, axis=1, keepdims=True), 1e-8)).T
    nearest = np.argmax(nn, axis=1)
    num_prediction_rows = test_pred.shape[0]
    if args.max_prediction_rows is not None:
        num_prediction_rows = min(num_prediction_rows, args.max_prediction_rows)
    with pred_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "range", "transition_type", "state_t", "action", "true_state_t1",
            "nearest_state_t1", "cosine_true", "nearest_is_exact_row",
        ])
        writer.writeheader()
        for i in range(num_prediction_rows):
            writer.writerow({
                "id": str(test["ids"][i]),
                "range": str(test["ranges"][i]),
                "transition_type": str(test["transition_types"][i]),
                "state_t": str(test["text_t"][i]),
                "action": str(test["action_text"][i]),
                "true_state_t1": str(test["text_t1"][i]),
                "nearest_state_t1": str(test["text_t1"][nearest[i]]),
                "cosine_true": float(test_cos[i]),
                "nearest_is_exact_row": bool(nearest[i] == i),
            })

    print(f"wrote {metrics_file}")
    print(f"wrote {pred_file}")
    del model, train, dev, test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def write_summary(all_metrics):
    path = RUN_DIR / "next_state_vector_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "model", "architecture", "embedding_dim", "num_train", "num_test", "best_epoch",
            "false_negative_mask",
            "test_cosine_mean", "test_cosine_median", "test_retrieval_top1",
            "test_retrieval_top5", "test_retrieval_top10", "test_mean_target_rank",
            "attribute_cosine_mean", "attribute_top1", "relation_cosine_mean", "relation_top1",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in all_metrics:
            attr = m["test_by_range"].get("attribute", {})
            rel = m["test_by_range"].get("relation", {})
            writer.writerow({
                "model": m["model"],
                "architecture": m["architecture"],
                "embedding_dim": m["embedding_dim"],
                "num_train": m["num_train"],
                "num_test": m["num_test"],
                "best_epoch": m["best_epoch"],
                "false_negative_mask": m.get("false_negative_mask", "none"),
                "test_cosine_mean": m["test"]["cosine_mean"],
                "test_cosine_median": m["test"]["cosine_median"],
                "test_retrieval_top1": m["test"]["retrieval_top1"],
                "test_retrieval_top5": m["test"]["retrieval_top5"],
                "test_retrieval_top10": m["test"]["retrieval_top10"],
                "test_mean_target_rank": m["test"]["mean_target_rank"],
                "attribute_cosine_mean": attr.get("cosine_mean"),
                "attribute_top1": attr.get("retrieval_top1"),
                "relation_cosine_mean": rel.get("cosine_mean"),
                "relation_top1": rel.get("retrieval_top1"),
            })
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--models-yaml", default=str(CONFIG))
    parser.add_argument("--data-prefix", default="next_state_vector")
    parser.add_argument("--experiment-name", default="next_state_vector")
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--overwrite-embeddings", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--architecture", choices=["concat", "film", "residual_film", "gated_residual_film", "pair_gated_residual", "state_only", "action_only"], default="residual_film")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--false-negative-mask", choices=["none", "same_next_text"], default="none")
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--max-prediction-rows", type=int, default=None)
    args = parser.parse_args()

    global OUT_DIR, EMB_DIR, RUN_DIR, ERROR_LOG, DATA_PREFIX
    DATA_PREFIX = args.data_prefix
    OUT_DIR = ROOT / "outputs" / args.experiment_name
    EMB_DIR = OUT_DIR / "embeddings"
    RUN_DIR = OUT_DIR / "runs"
    ERROR_LOG = OUT_DIR / "errors.log"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.no_generate:
        generate_data(args.seed, args.overwrite_data)
    models = load_models(args.models, args.models_yaml)
    if not args.skip_cache:
        for cfg in models:
            try:
                cache_embeddings(cfg, ["train", "dev", "test"], args.overwrite_embeddings, args.encode_batch_size)
            except Exception as exc:
                print(f"ERROR caching {cfg.get('name')}: {exc}", flush=True)
                log_error(f"cache:{cfg.get('name')}", exc)
                raise

    all_metrics = []
    for cfg in models:
        try:
            print(f"\n===== TRAIN NEXT-STATE VECTOR {cfg['name']} =====", flush=True)
            all_metrics.append(train_one(cfg, args))
        except Exception as exc:
            print(f"ERROR training {cfg.get('name')}: {exc}", flush=True)
            log_error(f"train:{cfg.get('name')}", exc)
            raise
    write_summary(all_metrics)


if __name__ == "__main__":
    main()
