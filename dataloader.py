#!/usr/bin/env python
"""Data loader for DEAM and MTG-Jamendo datasets."""

import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader

# MERT model cache
_MERT_MODEL_CACHE = {
    "dir": None,
    "processor": None,
    "model": None
}


def _apply_subset(items, subset_fraction: float = 1.0, subset_seed: int | None = None):
    """Return a deterministic subset of items.

    subset_fraction: float in (0, 1]. 1.0 means keep all.
    subset_seed: RNG seed for reproducibility.
    """
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


class DEAMDataset(Dataset):
    """DEAM Dataset with on-the-fly feature extraction."""

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
            base_dir,
            "data",
            "DEAM",
            "annotations",
            "annotations averaged per song",
            "song_level",
        )
        self.audio_root = audio_root or os.path.join(base_dir, "data", "DEAM", "audio")
        self.model_dir = model_dir

        # Load split IDs
        with open(os.path.join(base_dir, "data", "DEAM", "deam_split.json"), "r") as f:
            split_data = json.load(f)
            self.audio_ids = _apply_subset(split_data[self.split], subset_fraction, subset_seed)

        # Load labels
        self.labels = self._load_labels()

        # Initialize Mel transform with matched hop length
        self.mel_transform = T.MelSpectrogram(
            sample_rate=24000,  # Match MERT sample rate
            hop_length=320,      # 24000 / 75 = 320 for 75Hz frame rate
            n_fft=2048,          # Default reasonable value
            n_mels=128,          # Default reasonable value
            power=2.0
        )

    def _load_labels(self):
        """Load and combine DEAM labels from two CSV files."""
        labels = {}

        # Load first CSV (1-2000)
        csv_path1 = os.path.join(self.label_path, "static_annotations_averaged_songs_1_2000.csv")
        with open(csv_path1, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                song_id = parts[0]
                valence = float(parts[1])
                arousal = float(parts[3])
                labels[song_id] = (valence, arousal)

        # Load second CSV (2000-2058)
        csv_path2 = os.path.join(self.label_path, "static_annotations_averaged_songs_2000_2058.csv")
        with open(csv_path2, "r") as f:
            next(f)  # Skip header
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

        # Load audio file
        audio_path = os.path.join(self.audio_root, f"{audio_id}.mp3")
        waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension

        # Resample to MERT sample rate
        if sample_rate != 24000:
            resampler = T.Resample(sample_rate, 24000)
            waveform = resampler(waveform)

        # Extract 3 random 5-second segments
        segment_duration = 5  # 5 seconds per segment
        num_segments = 3
        segment_samples = segment_duration * 24000  # 5s * 24000Hz
        audio_length = waveform.shape[-1]

        # Initialize segments list
        segments = []

        for _ in range(num_segments):
            if audio_length >= segment_samples:
                # Randomly select a start point for the segment
                random_start = torch.randint(0, audio_length - segment_samples, (1,)).item()
                segments.append(waveform[:, random_start:random_start + segment_samples])
            else:
                # If audio is shorter than segment, pad it
                pad_length = segment_samples - audio_length
                padded = torch.nn.functional.pad(waveform, (0, pad_length))
                segments.append(padded)

        # Combine segments into one tensor
        waveform = torch.cat(segments, dim=1)  # [1, num_segments * segment_samples]

        # Extract MERT features
        processor, model = load_mert_model(self.model_dir)
        with torch.no_grad():
            inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=24000, return_tensors="pt", padding=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            mert = outputs.hidden_states[11].squeeze(0)  # Use only layer 11

        # Extract mel features
        mel = self.mel_transform(waveform.squeeze(0))  # [n_mels, Time]
        mel = torch.log(mel + 1e-9)  # Log mel

        # Apply SpecAugment (train split only)
        if self.split == "train":
            time_mask = T.TimeMasking(time_mask_param=30)
            mel = time_mask(mel)

            freq_mask = T.FrequencyMasking(freq_mask_param=20)
            mel = freq_mask(mel)

        mel = mel.transpose(1, 0)  # [Time, n_mels]

        # Extract chroma
        waveform_np = waveform.squeeze(0).numpy()
        chroma = librosa.feature.chroma_cqt(y=waveform_np, sr=24000, hop_length=320)
        chroma = torch.tensor(chroma, dtype=torch.float16).transpose(1, 0)  # [Time, D]

        # Extract tempogram
        tempogram = librosa.feature.tempogram(y=waveform_np, sr=24000, hop_length=320)
        tempogram = torch.tensor(tempogram, dtype=torch.float16).transpose(1, 0)  # [Time, D]

        # Get label
        valence, arousal = self.labels[audio_id]
        label = torch.tensor([valence, arousal], dtype=torch.float32)

        return {
            "mert": mert.to(torch.float16),
            "mel": mel.to(torch.float16),
            "chroma": chroma,
            "tempogram": tempogram
        }, label


class MTGJamendoDataset(Dataset):
    """MTG-Jamendo Dataset with on-the-fly feature extraction."""

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
        # Default to the curated subset CSV (generated from train/val/test folders)
        self.label_path = label_path or os.path.join(base_dir, "data", "MTG-Jamendo", "mtg_labels.csv")
        self.audio_root = audio_root or os.path.join(base_dir, "data", "MTG-Jamendo")
        self.model_dir = model_dir
        self.mood_tags = self._load_tags()
        # Avoid loading all rows into memory when using a subset.
        self.data = self._load_data(subset_fraction=subset_fraction, subset_seed=subset_seed)

        # Initialize Mel transform with matched hop length
        self.mel_transform = T.MelSpectrogram(
            sample_rate=24000,  # Match MERT sample rate
            hop_length=320,      # 24000 / 75 = 320 for 75Hz frame rate
            n_fft=2048,          # Default reasonable value
            n_mels=128,          # Default reasonable value
            power=2.0
        )

    def _load_tags(self):
        """Extract all unique mood tags."""
        tags = set()
        with open(self.label_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                mood_tag_list = parts[5].split("|")
                for tag in mood_tag_list:
                    tags.add(tag)
        return sorted(list(tags))

    def _load_data(self, subset_fraction: float = 1.0, subset_seed: int | None = None):
        """Load MTG-Jamendo data for the given split.

        If subset_fraction < 1, performs deterministic Bernoulli sampling while
        streaming the CSV to avoid holding the full split in memory.
        """
        # Validate subset fraction
        subset_fraction = float(subset_fraction) if subset_fraction is not None else 1.0
        if subset_fraction <= 0.0 or subset_fraction > 1.0:
            raise ValueError(f"subset_fraction must be in (0,1], got: {subset_fraction}")

        rng = np.random.default_rng(subset_seed)
        data = []
        first_candidate = None

        with open(self.label_path, "r") as f:
            next(f)  # Skip header
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

                # Skip if audio file doesn't exist
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

        # Avoid empty dataset when fraction is tiny
        if not data and first_candidate is not None:
            data = [first_candidate]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio_path"]

        # Load audio
        waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension

        # Resample
        if sample_rate != 24000:
            resampler = T.Resample(sample_rate, 24000)
            waveform = resampler(waveform)


        # Extract 3 random 5-second segments
        segment_duration = 5  # 5 seconds per segment
        num_segments = 3
        segment_samples = segment_duration * 24000  # 5s * 24000Hz
        audio_length = waveform.shape[-1]

        # Initialize segments list
        segments = []

        for _ in range(num_segments):
            if audio_length >= segment_samples:
                # Randomly select a start point for the segment
                random_start = torch.randint(0, audio_length - segment_samples, (1,)).item()
                segments.append(waveform[:, random_start:random_start + segment_samples])
            else:
                # If audio is shorter than segment, pad it
                pad_length = segment_samples - audio_length
                padded = torch.nn.functional.pad(waveform, (0, pad_length))
                segments.append(padded)

        # Combine segments into chunks
        chunks = torch.cat(segments, dim=1).unsqueeze(0)  # [1, 1, num_segments * segment_samples]

        # Extract MERT features
        processor, model = load_mert_model(self.model_dir)
        mert_features = []
        with torch.no_grad():
            for chunk in chunks:
                # Squeeze channel dim: [1, T]
                chunk = chunk.squeeze(1)
                inputs = processor(chunk.numpy(), sampling_rate=24000, return_tensors="pt", padding=False)

                # Move inputs to model device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model(**inputs, output_hidden_states=True)
                chunk_mert = outputs.hidden_states[11].squeeze(0)  # Use only layer 11 [T, D]
                mert_features.append(chunk_mert.cpu())
        mert = torch.cat(mert_features, dim=0)  # [T_total, D]

        # Extract mel features
        mel_chunks = []
        for chunk in chunks:
            # [1, T] -> [T]
            chunk = chunk.squeeze(0)
            mel = self.mel_transform(chunk)  # [n_mels, Time]
            mel = torch.log(mel + 1e-9)  # Log mel

            # Apply SpecAugment (train split only)
            if self.split == "train" and np.random.random() < 0.5:  # SpecAugment probability
                time_mask = T.TimeMasking(time_mask_param=30)
                mel = time_mask(mel)

                freq_mask = T.FrequencyMasking(freq_mask_param=20)
                mel = freq_mask(mel)

            mel = mel.transpose(1, 0)  # [Time, n_mels]
            mel_chunks.append(mel)
        mel = torch.cat(mel_chunks, dim=0)  # [T_total, D]

        # Extract chroma
        chroma_chunks = []
        for chunk in chunks:
            chunk_np = chunk.squeeze(0).numpy()
            chroma = librosa.feature.chroma_cqt(
                y=chunk_np,
                sr=24000,
                hop_length=320
            )
            chroma = torch.tensor(chroma).transpose(1, 0)  # [T, D]
            chroma_chunks.append(chroma)
        chroma = torch.cat(chroma_chunks, dim=0)  # [T_total, D]

        # Extract tempogram
        tempogram_chunks = []
        for chunk in chunks:
            chunk_np = chunk.squeeze(0).numpy()
            tempogram = librosa.feature.tempogram(
                y=chunk_np,
                sr=24000,
                hop_length=320
            )
            tempogram = torch.tensor(tempogram, dtype=torch.float32).transpose(1, 0)  # [T, D]
            tempogram_chunks.append(tempogram)
        tempogram = torch.cat(tempogram_chunks, dim=0)  # [T_total, D]

        # Encode labels
        mood_tags = item["mood_tags"]
        label = torch.zeros(len(self.mood_tags), dtype=torch.float32)
        for tag in mood_tags:
            if tag in self.mood_tags:
                label[self.mood_tags.index(tag)] = 1.0

        return {
            "mert": mert.to(torch.float16),
            "mel": mel.to(torch.float16),
            "chroma": chroma.to(torch.float16),
            "tempogram": tempogram.to(torch.float16)
        }, label


def collate_fn(batch):
    """Collate function to handle variable-length sequences."""
    # All features have the same shape across the batch in DEAM
    # For MTG-Jamendo, chunks are fixed to 30s, so total length varies based on number of chunks

    features_batch = {}
    labels = []

    # Separate features and labels
    all_features, all_labels = zip(*batch)

    # Find global max length across all feature types and all samples to ensure alignment
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
                padded = torch.nn.functional.pad(feat, (0, 0, 0, pad_len))  # Pad along time dimension
            else:
                padded = feat
            padded_features.append(padded)

        features_batch[key] = torch.stack(padded_features)

    # Stack labels
    labels = torch.stack(all_labels)

    return features_batch, labels


def get_dataloader(dataset_name, split="train", batch_size=8, shuffle=True, num_workers=0, **kwargs):
    """Get dataloader for the specified dataset."""
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
    # Test DEAM dataloader
    print("Testing DEAM Dataloader...")
    deam_loader = get_dataloader("deam", split="train", batch_size=4, model_dir="MERT")
    deam_batch = next(iter(deam_loader))
    deam_features, deam_labels = deam_batch
    print(f"DEAM Batch Features: {deam_features.keys()}")
    for key, val in deam_features.items():
        print(f"  {key}: {val.shape}")
    print(f"DEAM Batch Labels: {deam_labels.shape}")

    # Test MTG-Jamendo dataloader
    print("\nTesting MTG-Jamendo Dataloader...")
    try:
        mtg_loader = get_dataloader("mtg-jamendo", split="train", batch_size=2, model_dir="MERT", num_workers=0)
        mtg_batch = next(iter(mtg_loader))
        mtg_features, mtg_labels = mtg_batch
        print(f"MTG-Jamendo Batch Features: {mtg_features.keys()}")
        for key, val in mtg_features.items():
            print(f"  {key}: {val.shape}")
        print(f"MTG-Jamendo Batch Labels: {mtg_labels.shape}")
        print("MTG-Jamendo Dataloader test passed!")
    except Exception as e:
        print(f"MTG-Jamendo Dataloader test failed: {str(e)}")
