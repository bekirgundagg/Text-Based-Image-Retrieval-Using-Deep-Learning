"""CUB pilot: frozen ConvNeXt-Tiny/BERT, trained projection heads.
Run from the person_c directory. This is a validation pilot, not a final result.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from transformers import AutoTokenizer, BertModel


class DualEncoder(nn.Module):
    def __init__(self, dimension=256):
        super().__init__()
        self.image_encoder = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.image_encoder.classifier[2] = nn.Identity()
        self.text_encoder = BertModel.from_pretrained('google-bert/bert-base-uncased')
        self.image_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.image_projection = nn.Linear(768, dimension)
        self.text_projection = nn.Linear(768, dimension)

    def encode_image(self, features):
        return F.normalize(self.image_projection(features), dim=-1)

    def encode_text(self, features):
        return F.normalize(self.text_projection(features), dim=-1)


def recall_at_k(scores, targets, ks=(1, 5, 10)):
    """Each caption has one correct image; gallery columns are unique images."""
    ranking = scores.argsort(dim=1, descending=True)
    return {f'R@{k}': (ranking[:, :min(k, scores.shape[1])] == targets[:, None]).any(1).float().mean().item() for k in ks}


def build_records(root, seed):
    cub = root / 'CUB_200_2011'
    paths = dict(line.split(maxsplit=1) for line in (cub / 'images.txt').read_text().splitlines())
    flags = dict(line.split() for line in (cub / 'train_test_split.txt').read_text().splitlines())
    groups = {}
    for image_id, path in paths.items():
        if flags[image_id] == '1':
            groups.setdefault(path.split('/')[0], []).append((image_id, path))
    rng = random.Random(seed)
    records = {'train': [], 'val': []}
    for category in sorted(groups):
        entries = sorted(groups[category]); rng.shuffle(entries)
        n_val = max(1, round(len(entries) * .1))
        for position, (image_id, path) in enumerate(entries):
            captions = (root / 'birds/text' / Path(path).with_suffix('.txt')).read_text().splitlines()
            captions = [s.strip() for s in captions if s.strip()]
            if not captions: raise ValueError(f'Missing captions: {path}')
            records['val' if position < n_val else 'train'].append({'image_id': image_id, 'image_path': path, 'captions': captions})
    for rows in records.values(): rng.shuffle(rows)
    assert not ({r['image_id'] for r in records['train']} & {r['image_id'] for r in records['val']})
    return records


@torch.inference_mode()
def extract(model, tokenizer, rows, root, device, batch_size):
    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    image_features, text_features, owners = [], [], []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = []
        for row in batch:
            with Image.open(root / 'CUB_200_2011/images' / row['image_path']) as image:
                images.append(transform(image.convert('RGB')))
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            features = model.image_encoder(torch.stack(images).to(device))
        image_features.append(features.float().cpu())
    captions = []
    for i, row in enumerate(rows):
        captions.extend(row['captions']); owners.extend([i] * len(row['captions']))
    for start in range(0, len(captions), batch_size):
        tokens = tokenizer(captions[start:start + batch_size], padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            hidden = model.text_encoder(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        text_features.append(pooled.float().cpu())
    return torch.cat(image_features), torch.cat(text_features), torch.tensor(owners)


@torch.no_grad()
def evaluate(model, features, device):
    images, texts, owners = features
    gallery = model.encode_image(images.to(device))
    scores = model.encode_text(texts.to(device)) @ gallery.T
    return recall_at_k(scores.cpu(), owners)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('/content/person_c_data'))
    parser.add_argument('--output', type=Path, default=Path('/content/person_c_run'))
    parser.add_argument('--train-images', type=int, default=256)
    parser.add_argument('--val-images', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 2: raise ValueError('Contrastive training requires batch size >= 2')
    random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records = build_records(args.data, args.seed)
    full_counts = {k: len(v) for k, v in records.items()}
    for split, limit in [('train', args.train_images), ('val', args.val_images)]:
        if limit > 0: records[split] = records[split][:limit]
    manifest = json.dumps(records, sort_keys=True)
    (args.output / 'manifest.json').write_text(manifest)
    print('Split sizes:', {k: len(v) for k, v in records.items()}, flush=True)
    model = DualEncoder().to(device)
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    begin = time.perf_counter()
    features = {k: tuple(t.clone() for t in extract(model, tokenizer, rows, args.data, device, args.batch_size)) for k, rows in records.items()}
    extraction_seconds = time.perf_counter() - begin
    torch.save(features, args.output / 'features.pt')
    before = evaluate(model, features['val'], device)
    print('Validation before training:', before, flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=.01)
    train_images, train_texts, owners = features['train']
    choices = [(owners == i).nonzero().flatten().tolist() for i in range(len(train_images))]
    history, best = [], -1
    begin = time.perf_counter()
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_images)); total, count = 0., 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            if len(indices) < 2: continue
            texts = torch.tensor([random.choice(choices[i]) for i in indices.tolist()])
            logits = model.encode_image(train_images[indices].to(device)) @ model.encode_text(train_texts[texts].to(device)).T / .07
            targets = torch.arange(len(indices), device=device)
            loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(indices); count += len(indices)
        metrics = evaluate(model, features['val'], device)
        row = {'epoch': epoch + 1, 'loss': total / count, **metrics}; history.append(row)
        state = {'image_projection': model.image_projection.state_dict(), 'text_projection': model.text_projection.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'seed': args.seed, 'manifest_sha256': hashlib.sha256(manifest.encode()).hexdigest()}
        torch.save(state, args.output / 'last.pt')
        if metrics['R@1'] > best:
            best = metrics['R@1']; torch.save(state, args.output / 'best.pt')
        print(json.dumps(row), flush=True)
    elapsed = time.perf_counter() - begin
    result = {'status': 'validation_pilot_not_final_test', 'model': 'frozen ConvNeXt-Tiny + frozen BERT-base + trained 256D heads', 'seed': args.seed, 'train_images': len(train_images), 'val_images': len(records['val']), 'val_captions': len(features['val'][1]), 'full_split_counts': full_counts, 'before': before, 'best_validation': max(history, key=lambda r: r['R@1']), 'feature_extraction_seconds': extraction_seconds, 'training_seconds': elapsed, 'projection_epoch_seconds': elapsed / args.epochs, 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU', 'torch': torch.__version__, 'history': history}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    print('RESULT', json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)


if __name__ == '__main__':
    main()
