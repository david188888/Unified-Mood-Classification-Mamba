#!/usr/bin/env python
"""优化版数据加载器 - 支持预计算特征缓存，大幅加速训练。

主要优化:
1. 支持从磁盘加载预计算的 MERT/mel/chroma/tempogram 特征
2. 支持多进程数据加载 (num_workers > 0)
3. 训练时的数据增强（SpecAugment）在加载缓存特征后应用
"""

import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader

# MERT model cache (仅在非缓存模式下使用)
_MERT_MODEL_CACHE = {
    "dir": None,
    "processor": None,
    "model": None
}


def _apply_subset(items, subset_fraction: float = 1.0, subset_seed: int | None = None):
    """Return a deterministic subset of items."""
    if subset_fraction is None:
        return items

    try:
        subset_fraction = float(subset_fraction)
    except (TypeError, ValueError):
        raise ValueError(f"subset_fraction must be a float in (0,1], got: {subset_fraction!r}")

    if subset_fraction <= 0.0 or subset_fraction > 1.0:
        raise ValueError(f"subset_fraction must be in (0,1], got: {subset_fraction}")

    n = len(items)
    if n == 0 or subset_fraction == 1.0:
        return items

    k = max(1, int(round(n * subset_fraction)))
    rng = np.random.default_rng(subset_seed)
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    return [items[i] for i in idx]


def load_mert_model(model_dir):
    """Load (or reuse) the MERT model and processor."""
    global _MERT_MODEL_CACHE

    if _MERT_MODEL_CACHE["model"] is not None and _MERT_MODEL_CACHE["dir"] == model_dir:
        return _MERT_MODEL_CACHE["processor"], _MERT_MODEL_CACHE["model"]

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)

    if torch.backends.mps.is_available():
        model = model.to("mps")

    model.eval()
    _MERT_MODEL_CACHE = {
        "dir": model_dir,
        "processor": processor,
        "model": model
    }
    return processor, model


class DEAMDatasetCached(Dataset):
    """DEAM Dataset - 从预计算缓存加载特征（快速版）"""

    def __init__(
        self,
        split="train",
        label_path=None,
        cache_dir=None,
        subset_fraction: float = 1.0,
        subset_seed: int | None = None,
        apply_augment: bool = True,
    ):
        self.split = split
        self.apply_augment = apply_augment and (split == "train")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.label_path = label_path or os.path.join(
            base_dir, "data", "DEAM", "annotations",
            "annotations averaged per song", "song_level"
        )
        self.cache_dir = cache_dir or os.path.join(base_dir, "data", "features", "deam")

        # Load split IDs
        with open(os.path.join(base_dir, "data", "DEAM", "deam_split.json"), "r") as f:
            split_data = json.load(f)
            self.audio_ids = _apply_subset(split_data[self.split], subset_fraction, subset_seed)

        # 验证缓存文件是否存在
        valid_ids = []
        for audio_id in self.audio_ids:
            cache_path = os.path.join(self.cache_dir, f"{audio_id}.pt")
            if os.path.exists(cache_path):
                valid_ids.append(audio_id)
            else:
                print(f"警告: 缓存文件不存在 {cache_path}")
        self.audio_ids = valid_ids

        # Load labels
        self.labels = self._load_labels()

        # SpecAugment transforms
        if self.apply_augment:
            self.time_mask = T.TimeMasking(time_mask_param=30)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=20)

    def _load_labels(self):
        """Load and combine DEAM labels from two CSV files."""
        labels = {}
        
        csv_path1 = os.path.join(self.label_path, "static_annotations_averaged_songs_1_2000.csv")
        with open(csv_path1, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                song_id = parts[0]
                valence = float(parts[1])
                arousal = float(parts[3])
                labels[song_id] = (valence, arousal)

        csv_path2 = os.path.join(self.label_path, "static_annotations_averaged_songs_2000_2058.csv")
        with open(csv_path2, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                song_id = parts[0]
                valence = float(parts[1])
                arousal = float(parts[3])
                labels[song_id] = (valence, arousal)

        return labels

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, idx):
        audio_id = self.audio_ids[idx]
        
        # 加载预计算的特征
        cache_path = os.path.join(self.cache_dir, f"{audio_id}.pt")
        features = torch.load(cache_path, weights_only=True)
        
        mert = features['mert']
        mel = features['mel']
        chroma = features['chroma']
        tempogram = features['tempogram']
        
        # 应用 SpecAugment (仅训练集)
        if self.apply_augment:
            # mel: [Time, n_mels] -> [n_mels, Time] for augment
            mel_t = mel.transpose(0, 1).float()
            mel_t = self.time_mask(mel_t)
            mel_t = self.freq_mask(mel_t)
            mel = mel_t.transpose(0, 1).to(torch.float16)

        # Get label
        valence, arousal = self.labels[audio_id]
        label = torch.tensor([valence, arousal], dtype=torch.float32)

        return {
            "mert": mert,
            "mel": mel,
            "chroma": chroma,
            "tempogram": tempogram
        }, label


class MTGJamendoDatasetCached(Dataset):
    """MTG-Jamendo Dataset - 从预计算缓存加载特征（快速版）"""

    def __init__(
        self,
        split="train",
        label_path=None,
        cache_dir=None,
        subset_fraction: float = 1.0,
        subset_seed: int | None = None,
        apply_augment: bool = True,
    ):
        self.split = split
        self.apply_augment = apply_augment and (split == "train")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Default to curated subset CSV
        self.label_path = label_path or os.path.join(base_dir, "data", "MTG-Jamendo", "mtg_labels.csv")
        self.cache_dir = cache_dir or os.path.join(base_dir, "data", "features", "mtg")
        
        self.mood_tags = self._load_tags()
        self.data = self._load_data(subset_fraction=subset_fraction, subset_seed=subset_seed)

        # SpecAugment transforms
        if self.apply_augment:
            self.time_mask = T.TimeMasking(time_mask_param=30)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=20)

    def _load_tags(self):
        """Extract all unique mood tags."""
        tags = set()
        with open(self.label_path, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                mood_tag_list = parts[5].split("|")
                for tag in mood_tag_list:
                    tags.add(tag)
        return sorted(list(tags))

    def _load_data(self, subset_fraction: float = 1.0, subset_seed: int | None = None):
        """Load MTG-Jamendo data for the given split."""
        subset_fraction = float(subset_fraction) if subset_fraction is not None else 1.0
        if subset_fraction <= 0.0 or subset_fraction > 1.0:
            raise ValueError(f"subset_fraction must be in (0,1], got: {subset_fraction}")

        rng = np.random.default_rng(subset_seed)
        data = []
        first_candidate = None

        with open(self.label_path, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                split, track_id, _, _, _, mood_tags_str, audio_path_rel = parts
                track_id = int(track_id)
                if split != self.split:
                    continue

                # 使用 split/track_id 格式的缓存路径 (e.g., train/948.pt)
                cache_path = os.path.join(self.cache_dir, split, f"{track_id}.pt")
                if not os.path.exists(cache_path):
                    continue

                mood_tags = mood_tags_str.split("|")
                item = {
                    "track_id": track_id,
                    "cache_path": cache_path,
                    "mood_tags": mood_tags,
                }

                if first_candidate is None:
                    first_candidate = item

                if subset_fraction < 1.0 and rng.random() >= subset_fraction:
                    continue

                data.append(item)

        if not data and first_candidate is not None:
            data = [first_candidate]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载预计算的特征
        features = torch.load(item["cache_path"], weights_only=True)
        
        mert = features['mert']
        mel = features['mel']
        chroma = features['chroma']
        tempogram = features['tempogram']
        
        # 应用 SpecAugment (仅训练集，50% 概率)
        if self.apply_augment and np.random.random() < 0.5:
            mel_t = mel.transpose(0, 1).float()
            mel_t = self.time_mask(mel_t)
            mel_t = self.freq_mask(mel_t)
            mel = mel_t.transpose(0, 1).to(torch.float16)

        # Encode labels
        label = torch.zeros(len(self.mood_tags), dtype=torch.float32)
        for tag in item["mood_tags"]:
            if tag in self.mood_tags:
                label[self.mood_tags.index(tag)] = 1.0

        return {
            "mert": mert,
            "mel": mel,
            "chroma": chroma,
            "tempogram": tempogram
        }, label


def collate_fn(batch):
    """Collate function to handle variable-length sequences."""
    features_batch = {}
    labels = []

    all_features, all_labels = zip(*batch)

    # Find global max length
    max_len = 0
    for feat_dict in all_features:
        for feat in feat_dict.values():
            if feat.shape[0] > max_len:
                max_len = feat.shape[0]

    # Stack each feature type
    for key in all_features[0].keys():
        features = [feat[key] for feat in all_features]
        
        padded_features = []
        for feat in features:
            pad_len = max_len - feat.shape[0]
            if pad_len > 0:
                padded = torch.nn.functional.pad(feat, (0, 0, 0, pad_len))
            else:
                padded = feat
            padded_features.append(padded)

        features_batch[key] = torch.stack(padded_features)

    labels = torch.stack(all_labels)

    return features_batch, labels


def get_dataloader_fast(dataset_name, split="train", batch_size=8, shuffle=True, 
                        num_workers=4, prefetch_factor=2, pin_memory=True, **kwargs):
    """获取优化后的数据加载器（使用预计算特征）
    
    Args:
        dataset_name: 'deam' 或 'mtg-jamendo'
        split: 'train', 'val', 或 'test'
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 数据加载进程数（推荐 2-4）
        prefetch_factor: 每个 worker 预取的批次数
        pin_memory: 是否将数据固定在内存（加速 GPU 传输）
        **kwargs: 其他参数传递给 Dataset
    """
    if dataset_name == "deam":
        dataset = DEAMDatasetCached(split=split, **kwargs)
    elif dataset_name == "mtg-jamendo":
        dataset = MTGJamendoDatasetCached(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 对于 macOS，num_workers > 0 时使用 spawn 模式可能有问题
    # 但加载预计算特征通常很快，所以 num_workers=0 也可接受
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': pin_memory if torch.cuda.is_available() else False,
        'persistent_workers': num_workers > 0,
    }
    
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


# ============================================================================
# 兼容旧接口 - 实时特征提取版本（保留原始功能）
# ============================================================================

class DEAMDataset(Dataset):
    """DEAM Dataset with on-the-fly feature extraction (原始版本)"""

    def __init__(
        self,
        split="train",
        label_path=None,
        audio_root=None,
        model_dir=None,
        subset_fraction: float = 1.0,
        subset_seed: int | None = None,
    ):
        self.split = split
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.label_path = label_path or os.path.join(
            base_dir, "data", "DEAM", "annotations",
            "annotations averaged per song", "song_level",
        )
        self.audio_root = audio_root or os.path.join(base_dir, "data", "DEAM", "audio")
        self.model_dir = model_dir

        with open(os.path.join(base_dir, "data", "DEAM", "deam_split.json"), "r") as f:
            split_data = json.load(f)
            self.audio_ids = _apply_subset(split_data[self.split], subset_fraction, subset_seed)

        self.labels = self._load_labels()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=24000, hop_length=320, n_fft=2048, n_mels=128, power=2.0
        )

    def _load_labels(self):
        labels = {}
        csv_path1 = os.path.join(self.label_path, "static_annotations_averaged_songs_1_2000.csv")
        with open(csv_path1, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                song_id = parts[0]
                valence = float(parts[1])
                arousal = float(parts[3])
                labels[song_id] = (valence, arousal)

        csv_path2 = os.path.join(self.label_path, "static_annotations_averaged_songs_2000_2058.csv")
        with open(csv_path2, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                song_id = parts[0]
                valence = float(parts[1])
                arousal = float(parts[3])
                labels[song_id] = (valence, arousal)
        return labels

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, idx):
        audio_id = self.audio_ids[idx]
        audio_path = os.path.join(self.audio_root, f"{audio_id}.mp3")
        waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.tensor(waveform).unsqueeze(0)

        if sample_rate != 24000:
            resampler = T.Resample(sample_rate, 24000)
            waveform = resampler(waveform)

        segment_duration = 5
        num_segments = 3
        segment_samples = segment_duration * 24000
        audio_length = waveform.shape[-1]

        segments = []
        for _ in range(num_segments):
            if audio_length >= segment_samples:
                random_start = torch.randint(0, audio_length - segment_samples, (1,)).item()
                segments.append(waveform[:, random_start:random_start + segment_samples])
            else:
                pad_length = segment_samples - audio_length
                padded = torch.nn.functional.pad(waveform, (0, pad_length))
                segments.append(padded)

        waveform = torch.cat(segments, dim=1)

        processor, model = load_mert_model(self.model_dir)
        with torch.no_grad():
            inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=24000, return_tensors="pt", padding=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            mert = outputs.hidden_states[11].squeeze(0)

        mel = self.mel_transform(waveform.squeeze(0))
        mel = torch.log(mel + 1e-9)

        if self.split == "train":
            time_mask = T.TimeMasking(time_mask_param=30)
            mel = time_mask(mel)
            freq_mask = T.FrequencyMasking(freq_mask_param=20)
            mel = freq_mask(mel)

        mel = mel.transpose(1, 0)

        waveform_np = waveform.squeeze(0).numpy()
        chroma = librosa.feature.chroma_cqt(y=waveform_np, sr=24000, hop_length=320)
        chroma = torch.tensor(chroma, dtype=torch.float16).transpose(1, 0)

        tempogram = librosa.feature.tempogram(y=waveform_np, sr=24000, hop_length=320)
        tempogram = torch.tensor(tempogram, dtype=torch.float16).transpose(1, 0)

        valence, arousal = self.labels[audio_id]
        label = torch.tensor([valence, arousal], dtype=torch.float32)

        return {
            "mert": mert.to(torch.float16),
            "mel": mel.to(torch.float16),
            "chroma": chroma,
            "tempogram": tempogram
        }, label


class MTGJamendoDataset(Dataset):
    """MTG-Jamendo Dataset with on-the-fly feature extraction (原始版本)"""

    def __init__(
        self,
        split="train",
        label_path=None,
        audio_root=None,
        model_dir=None,
        subset_fraction: float = 1.0,
        subset_seed: int | None = None,
    ):
        self.split = split
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Default to curated subset CSV
        self.label_path = label_path or os.path.join(base_dir, "data", "MTG-Jamendo", "mtg_labels.csv")
        self.audio_root = audio_root or os.path.join(base_dir, "data", "MTG-Jamendo")
        self.model_dir = model_dir
        self.mood_tags = self._load_tags()
        self.data = self._load_data(subset_fraction=subset_fraction, subset_seed=subset_seed)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=24000, hop_length=320, n_fft=2048, n_mels=128, power=2.0
        )

    def _load_tags(self):
        tags = set()
        with open(self.label_path, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                mood_tag_list = parts[5].split("|")
                for tag in mood_tag_list:
                    tags.add(tag)
        return sorted(list(tags))

    def _load_data(self, subset_fraction: float = 1.0, subset_seed: int | None = None):
        subset_fraction = float(subset_fraction) if subset_fraction is not None else 1.0
        if subset_fraction <= 0.0 or subset_fraction > 1.0:
            raise ValueError(f"subset_fraction must be in (0,1], got: {subset_fraction}")

        rng = np.random.default_rng(subset_seed)
        data = []
        first_candidate = None

        with open(self.label_path, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                split, track_id, _, _, _, mood_tags_str, audio_path_rel = parts
                track_id = int(track_id)
                if split != self.split:
                    continue

                # Use audio_path from CSV (e.g., train/track_0000948.low.mp3)
                audio_path = os.path.join(self.audio_root, audio_path_rel)

                if not os.path.exists(audio_path):
                    continue

                mood_tags = mood_tags_str.split("|")
                item = {
                    "track_id": track_id,
                    "audio_path": audio_path,
                    "mood_tags": mood_tags,
                }

                if first_candidate is None:
                    first_candidate = item

                if subset_fraction < 1.0 and rng.random() >= subset_fraction:
                    continue

                data.append(item)

        if not data and first_candidate is not None:
            data = [first_candidate]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio_path"]

        waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.tensor(waveform).unsqueeze(0)

        if sample_rate != 24000:
            resampler = T.Resample(sample_rate, 24000)
            waveform = resampler(waveform)

        segment_duration = 5
        num_segments = 3
        segment_samples = segment_duration * 24000
        audio_length = waveform.shape[-1]

        segments = []
        for _ in range(num_segments):
            if audio_length >= segment_samples:
                random_start = torch.randint(0, audio_length - segment_samples, (1,)).item()
                segments.append(waveform[:, random_start:random_start + segment_samples])
            else:
                pad_length = segment_samples - audio_length
                padded = torch.nn.functional.pad(waveform, (0, pad_length))
                segments.append(padded)

        chunks = torch.cat(segments, dim=1).unsqueeze(0)

        processor, model = load_mert_model(self.model_dir)
        mert_features = []
        with torch.no_grad():
            for chunk in chunks:
                chunk = chunk.squeeze(1)
                inputs = processor(chunk.numpy(), sampling_rate=24000, return_tensors="pt", padding=False)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                chunk_mert = outputs.hidden_states[11].squeeze(0)
                mert_features.append(chunk_mert.cpu())
        mert = torch.cat(mert_features, dim=0)

        mel_chunks = []
        for chunk in chunks:
            chunk = chunk.squeeze(0)
            mel = self.mel_transform(chunk)
            mel = torch.log(mel + 1e-9)

            if self.split == "train" and np.random.random() < 0.5:
                time_mask = T.TimeMasking(time_mask_param=30)
                mel = time_mask(mel)
                freq_mask = T.FrequencyMasking(freq_mask_param=20)
                mel = freq_mask(mel)

            mel = mel.transpose(1, 0)
            mel_chunks.append(mel)
        mel = torch.cat(mel_chunks, dim=0)

        chroma_chunks = []
        for chunk in chunks:
            chunk_np = chunk.squeeze(0).numpy()
            chroma = librosa.feature.chroma_cqt(y=chunk_np, sr=24000, hop_length=320)
            chroma = torch.tensor(chroma).transpose(1, 0)
            chroma_chunks.append(chroma)
        chroma = torch.cat(chroma_chunks, dim=0)

        tempogram_chunks = []
        for chunk in chunks:
            chunk_np = chunk.squeeze(0).numpy()
            tempogram = librosa.feature.tempogram(y=chunk_np, sr=24000, hop_length=320)
            tempogram = torch.tensor(tempogram, dtype=torch.float32).transpose(1, 0)
            tempogram_chunks.append(tempogram)
        tempogram = torch.cat(tempogram_chunks, dim=0)

        label = torch.zeros(len(self.mood_tags), dtype=torch.float32)
        for tag in item["mood_tags"]:
            if tag in self.mood_tags:
                label[self.mood_tags.index(tag)] = 1.0

        return {
            "mert": mert.to(torch.float16),
            "mel": mel.to(torch.float16),
            "chroma": chroma.to(torch.float16),
            "tempogram": tempogram.to(torch.float16)
        }, label


def get_dataloader(dataset_name, split="train", batch_size=8, shuffle=True, num_workers=0, **kwargs):
    """Get dataloader for the specified dataset (原始版本，实时提取特征)"""
    if dataset_name == "deam":
        dataset = DEAMDataset(split=split, **kwargs)
    elif dataset_name == "mtg-jamendo":
        if "model_dir" not in kwargs:
            raise ValueError("model_dir is required for mtg-jamendo dataset")
        dataset = MTGJamendoDataset(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("测试优化版数据加载器 (使用预计算特征)")
    print("=" * 60)
    
    # 检查缓存目录是否存在
    base_dir = os.path.dirname(os.path.abspath(__file__))
    deam_cache = os.path.join(base_dir, "data", "features", "deam")
    mtg_cache = os.path.join(base_dir, "data", "features", "mtg")
    
    if not os.path.exists(deam_cache) or not os.path.exists(mtg_cache):
        print("\n⚠️  缓存目录不存在！请先运行预计算脚本：")
        print("   python precompute_features.py --dataset all")
        print("\n将使用原始数据加载器进行测试...")
        
        # 测试原始 DEAM dataloader
        print("\n测试 DEAM Dataloader (原始版本)...")
        deam_loader = get_dataloader("deam", split="train", batch_size=2, model_dir="MERT")
        deam_batch = next(iter(deam_loader))
        deam_features, deam_labels = deam_batch
        print(f"DEAM Batch Features: {deam_features.keys()}")
        for key, val in deam_features.items():
            print(f"  {key}: {val.shape}")
        print(f"DEAM Batch Labels: {deam_labels.shape}")
    else:
        # 测试优化版 DEAM dataloader
        print("\n测试 DEAM Dataloader (优化版)...")
        deam_loader = get_dataloader_fast("deam", split="train", batch_size=4, num_workers=0)
        deam_batch = next(iter(deam_loader))
        deam_features, deam_labels = deam_batch
        print(f"DEAM Batch Features: {deam_features.keys()}")
        for key, val in deam_features.items():
            print(f"  {key}: {val.shape}, dtype={val.dtype}")
        print(f"DEAM Batch Labels: {deam_labels.shape}")
        
        # 测试优化版 MTG-Jamendo dataloader
        print("\n测试 MTG-Jamendo Dataloader (优化版)...")
        mtg_loader = get_dataloader_fast("mtg-jamendo", split="train", batch_size=4, num_workers=0)
        mtg_batch = next(iter(mtg_loader))
        mtg_features, mtg_labels = mtg_batch
        print(f"MTG-Jamendo Batch Features: {mtg_features.keys()}")
        for key, val in mtg_features.items():
            print(f"  {key}: {val.shape}, dtype={val.dtype}")
        print(f"MTG-Jamendo Batch Labels: {mtg_labels.shape}")
        
        print("\n✅ 优化版数据加载器测试通过!")
