#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
은하 분류를 위한 사진 인식 프로그램 (외부 API/네트워크 미사용, GUI 포함)
- 프레임워크: PyTorch + torchvision + matplotlib + tkinter (로컬 전용)
- 데이터 구조 자동 인식(ImageFolder 규칙):
  dataset/
    train/
      Spiral/
        img001.jpg
        ...
      Elliptical/
        ...
      Irregular/
        ...
    val/            # 선택(없으면 train에서 자동 분할)
    test/           # 선택(있으면 평가 시 사용)

- 제공 기능
  1) 학습 (train): 학습곡선(log/acc) 저장, PNG/JSON/HTML 리포트 생성
  2) 평가 (eval): 정확도/혼동행렬/클래스별 F1
  3) 예측 (predict): 단일 이미지 또는 폴더, "항목별(클래스별) 확률" 상세 출력 및 JSON 저장
  4) GUI (gui): 체크포인트 로드 후 이미지 열기/폴더 탐색, 예측 결과 + 클래스별 확률 바 차트 시각화

- 체크포인트 저장: outputs/best_model.pt
- 라벨 매핑 저장: outputs/labels.json
- 학습 로그 저장: outputs/training_log.json
- 학습 곡선: outputs/curve_loss.png, outputs/curve_acc.png
- 공유용 리포트: outputs/report.html (이미지 Base64 임베드)

사용 예시:
  학습:   python galaxy_classifier.py --mode train --data_dir ./dataset --epochs 30
  평가:   python galaxy_classifier.py --mode eval --data_dir ./dataset --checkpoint outputs/best_model.pt
  예측:   python galaxy_classifier.py --mode predict --checkpoint outputs/best_model.pt --input ./some_image.jpg --show_probs
  폴더예측: python galaxy_classifier.py --mode predict --checkpoint outputs/best_model.pt --input ./unseen_folder --show_probs
  GUI:    python galaxy_classifier.py --mode gui --checkpoint outputs/best_model.pt
"""

import os
import sys
import io
import json
import time
import base64
import math
import copy
import random
import argparse
from typing import Dict, List, Tuple, Optional

# ---- ML Core ----
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, resnet34

from PIL import Image, ImageTk

# ---- Plotting ----
import matplotlib
# GUI 없는 서버 학습에서도 안전하게 작동하도록 Agg 백엔드 기본 지정
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
def setup_korean_font():
    """matplotlib 한글 폰트 설정"""
    try:
        # Windows
        if sys.platform.startswith('win'):
            plt.rcParams['font.family'] = 'Malgun Gothic'
        # macOS
        elif sys.platform.startswith('darwin'):
            plt.rcParams['font.family'] = 'AppleGothic'
        # Linux
        else:
            plt.rcParams['font.family'] = 'NanumGothic'
        
        # 마이너스 기호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 폰트 설정 실패 시 기본 설정 사용
        plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# ---- GUI (tkinter) ----
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# =====================
# 유틸리티
# =====================

# 지원하는 이미지 확장자 목록
SUPPORTED_IMAGE_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.bmp', 
    '.tif', '.tiff', '.webp', '.gif', 
    '.svg', '.ico', '.jfif', '.pjpeg', '.pjp'
)

def is_image_file(filename: str) -> bool:
    """파일이 지원되는 이미지 형식인지 확인"""
    return filename.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def img_to_base64(png_path: str) -> str:
    with open(png_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


# =====================
# 데이터셋 & 전처리
# =====================

def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_datasets(data_dir: str, img_size: int = 224, val_split: float = 0.15, seed: int = 42):
    set_seed(seed)
    train_tf, eval_tf = build_transforms(img_size)

    train_dir = os.path.join(data_dir, 'train')
    if not os.path.isdir(train_dir):
        # train 폴더가 없으면 경고 메시지와 함께 예외 발생
        # (데이터가 있어야 하므로 자동 생성하지 않음)
        raise FileNotFoundError(
            f"train 디렉터리가 필요합니다: {train_dir}\n"
            f"데이터셋 구조: {data_dir}/train/클래스명/이미지파일들"
        )

    full_train = ImageFolder(train_dir, transform=train_tf)

    val_dir = os.path.join(data_dir, 'val')
    if os.path.isdir(val_dir):
        val_set = ImageFolder(val_dir, transform=eval_tf)
    else:
        indices = list(range(len(full_train)))
        random.shuffle(indices)
        split = int(len(indices) * (1 - val_split))
        train_idx, val_idx = indices[:split], indices[split:]
        val_set = copy.deepcopy(full_train)
        val_set.transform = eval_tf
        val_set = Subset(val_set, val_idx)
        full_train = Subset(full_train, train_idx)

    test_dir = os.path.join(data_dir, 'test')
    test_set = ImageFolder(test_dir, transform=eval_tf) if os.path.isdir(test_dir) else None

    if isinstance(full_train, Subset):
        class_to_idx = full_train.dataset.class_to_idx
        classes = full_train.dataset.classes
    else:
        class_to_idx = full_train.class_to_idx
        classes = full_train.classes

    return full_train, val_set, test_set, classes, class_to_idx


# =====================
# 모델 구축
# =====================

def build_model(num_classes: int, backbone: str = 'resnet18') -> nn.Module:
    if backbone == 'resnet18':
        net = resnet18(weights=None)
    elif backbone == 'resnet34':
        net = resnet34(weights=None)
    else:
        raise ValueError('지원하지 않는 백본입니다: ' + backbone)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    return net


# =====================
# 학습/평가 루프
# =====================

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def compute_confusion_matrix(all_targets: List[int], all_preds: List[int], num_classes: int) -> List[List[int]]:
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(all_targets, all_preds):
        cm[t][p] += 1
    return cm


def per_class_precision_recall_f1(cm: List[List[int]]) -> Tuple[List[float], List[float], List[float]]:
    num_classes = len(cm)
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes) if r != c)
        fn = sum(cm[c][r] for r in range(num_classes) if r != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip: float = 1.0):
    """
    1 에포크 학습 수행
    
    Args:
        model: 학습할 모델
        loader: 학습 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        scaler: 혼합 정밀도 학습을 위한 GradScaler (None이면 일반 학습)
        device: 학습 장치 (cuda/cpu)
        grad_clip: 그래디언트 클리핑 임계값
    
    Returns:
        epoch_loss: 에포크 평균 손실
        epoch_acc: 에포크 평균 정확도
    """
    model.train()  # 모델을 학습 모드로 설정 (dropout, batchnorm 활성화)
    epoch_loss = 0.0
    epoch_acc = 0.0
    total = 0
    
    # 배치 단위로 학습 진행
    for images, targets in loader:
        # 데이터를 GPU/CPU로 이동
        images = images.to(device)
        targets = targets.to(device)
        
        # 이전 그래디언트 초기화 (메모리 효율을 위해 set_to_none=True 사용)
        optimizer.zero_grad(set_to_none=True)
        
        # 혼합 정밀도 학습 적용 (GPU 메모리 절약 및 속도 향상)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(images)  # 순전파: 예측값 계산
            loss = criterion(logits, targets)  # 손실 계산
        
        # 혼합 정밀도 학습 사용 시
        if scaler is not None:
            scaler.scale(loss).backward()  # 역전파: 그래디언트 계산 (스케일링 적용)
            
            # 그래디언트 클리핑 (폭발하는 그래디언트 방지)
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)  # 그래디언트 스케일 해제
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)  # 가중치 업데이트
            scaler.update()  # scaler 상태 업데이트
        
        # 일반 학습 (혼합 정밀도 미사용)
        else:
            loss.backward()  # 역전파
            
            # 그래디언트 클리핑
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()  # 가중치 업데이트
        
        # 배치 정확도 계산
        batch_acc = accuracy_from_logits(logits.detach(), targets)
        bs = targets.size(0)
        
        # 누적 손실 및 정확도 계산
        epoch_loss += loss.item() * bs
        epoch_acc += batch_acc * bs
        total += bs
    
    # 평균 손실 및 정확도 반환
    return epoch_loss / total, epoch_acc / total


essential_eval_keys = ['loss', 'acc', 'cm', 'per_class_precision', 'per_class_recall', 'per_class_f1']

def evaluate(model, loader, criterion, device, num_classes: int) -> Dict[str, object]:
    """
    모델 평가 수행
    
    Args:
        model: 평가할 모델
        loader: 평가 데이터 로더
        criterion: 손실 함수
        device: 평가 장치 (cuda/cpu)
        num_classes: 클래스 개수
    
    Returns:
        평가 지표 딕셔너리:
            - loss: 평균 손실
            - acc: 평균 정확도
            - cm: 혼동 행렬 (confusion matrix)
            - per_class_precision: 클래스별 정밀도
            - per_class_recall: 클래스별 재현율
            - per_class_f1: 클래스별 F1 점수
    """
    model.eval()  # 모델을 평가 모드로 설정 (dropout 비활성화, batchnorm frozen)
    epoch_loss = 0.0
    epoch_acc = 0.0
    total = 0
    all_targets: List[int] = []  # 실제 정답 레이블
    all_preds: List[int] = []    # 예측 레이블
    
    # 그래디언트 계산 비활성화 (메모리 절약 및 속도 향상)
    with torch.no_grad():
        for images, targets in loader:
            # 데이터를 GPU/CPU로 이동
            images = images.to(device)
            targets = targets.to(device)
            
            # 순전파: 예측값 계산
            logits = model(images)
            loss = criterion(logits, targets)
            
            # 가장 높은 확률의 클래스를 예측값으로 선택
            preds = torch.argmax(logits, dim=1)
            
            bs = targets.size(0)
            
            # 누적 손실 및 정확도 계산
            epoch_loss += loss.item() * bs
            epoch_acc += (preds == targets).sum().item()
            total += bs
            
            # 혼동 행렬 계산을 위해 예측값과 실제값 저장
            all_targets.extend(targets.tolist())
            all_preds.extend(preds.tolist())
    
    # 혼동 행렬 계산
    cm = compute_confusion_matrix(all_targets, all_preds, num_classes)
    
    # 클래스별 정밀도, 재현율, F1 점수 계산
    prec, rec, f1 = per_class_precision_recall_f1(cm)
    
    return {
        'loss': epoch_loss / total,
        'acc': epoch_acc / total,
        'cm': cm,
        'per_class_precision': prec,
        'per_class_recall': rec,
        'per_class_f1': f1,
    }


# =====================
# 학습 곡선/리포트
# =====================

def plot_training_curves(history: Dict[str, List[float]], out_dir: str):
    ensure_dir(out_dir)
    # 한글 폰트 설정
    setup_korean_font()
    
    # Loss
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    loss_path = os.path.join(out_dir, 'curve_loss.png')
    plt.savefig(loss_path, bbox_inches='tight')
    plt.close()
    # Acc
    plt.figure()
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    acc_path = os.path.join(out_dir, 'curve_acc.png')
    plt.savefig(acc_path, bbox_inches='tight')
    plt.close()
    return loss_path, acc_path


def save_html_report(out_dir: str, best_acc: float, last_val: Dict[str, object]):
    loss_png = os.path.join(out_dir, 'curve_loss.png')
    acc_png = os.path.join(out_dir, 'curve_acc.png')
    loss_b64 = img_to_base64(loss_png) if os.path.isfile(loss_png) else ''
    acc_b64 = img_to_base64(acc_png) if os.path.isfile(acc_png) else ''

    html = f"""
<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>은하 분류 학습 리포트</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;max-width:960px;margin:2rem auto;}}
.card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:16px;}}
pre{{white-space:pre-wrap;}}</style></head><body>
<h1>은하 분류 학습 리포트</h1>
<div class="card"><h2>최고 검증 정확도</h2><p>{best_acc:.4f}</p></div>
<div class="card"><h2>최종 검증 지표</h2>
<pre>{json.dumps(last_val, ensure_ascii=False, indent=2)}</pre></div>
<div class="card"><h2>학습 곡선 (Loss)</h2>
<img alt="loss" src="data:image/png;base64,{loss_b64}" style="max-width:100%"></div>
<div class="card"><h2>학습 곡선 (Accuracy)</h2>
<img alt="acc" src="data:image/png;base64,{acc_b64}" style="max-width:100%"></div>
<p style="color:#666">이 HTML 파일은 로컬에서 바로 열어 공유할 수 있습니다(이미지 Base64 포함).</p>
</body></html>
"""
    with open(os.path.join(out_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html)


# =====================
# 학습 드라이버
# =====================

def train_driver(args):
    """
    학습 전체 프로세스 실행
    
    Args:
        args: 명령줄 인자 (데이터 경로, 하이퍼파라미터 등)
    """
    # 재현성을 위한 시드 설정
    set_seed(args.seed)
    
    # 장치 설정 (GPU 사용 가능하면 GPU, 아니면 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # 데이터셋 로드 및 전처리
    train_set, val_set, test_set, classes, class_to_idx = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    # 데이터 로더 생성 (배치 단위로 데이터 로딩)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 모델 생성 및 GPU/CPU로 이동
    model = build_model(num_classes=len(classes), backbone=args.backbone)
    model.to(device)

    # 손실 함수, 옵티마이저, 스케줄러 설정
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류를 위한 크로스 엔트로피 손실
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # AdamW 옵티마이저
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # 학습률 스케줄러 (코사인 어닐링)
    
    # 혼합 정밀도 학습 설정 (GPU 사용 시 활성화 가능)
    scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda' and args.amp) else None

    # 최고 성능 모델 추적 변수
    best_acc = -1.0
    best_state = None

    # 출력 디렉터리 생성
    ensure_dir(args.output_dir)

    # 클래스 레이블 정보 저장 (예측 시 사용)
    labels_path = os.path.join(args.output_dir, 'labels.json')
    save_json({'classes': classes, 'class_to_idx': class_to_idx}, labels_path)

    # 학습 히스토리 초기화 (손실 및 정확도 추적)
    history = { 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [] }

    # 에포크별 학습 루프
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # 1 에포크 학습 수행
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_clip=args.grad_clip)
        
        # 검증 데이터로 모델 평가
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes=len(classes))
        
        # 학습률 스케줄러 업데이트
        scheduler.step()

        # 히스토리에 기록
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])

        # 진행 상황 출력
        elapsed = time.time() - t0
        print(f"[Epoch {epoch:03d}/{args.epochs}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} | time={elapsed:.1f}s")

        # 최고 검증 정확도 달성 시 모델 저장
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            best_state = {
                'epoch': epoch,
                'model_state': copy.deepcopy(model.state_dict()),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'classes': classes,
                'class_to_idx': class_to_idx,
                'args': vars(args),
                'val_metrics': val_metrics,
            }
            ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(best_state, ckpt_path)
            print(f"  -> 베스트 모델 저장: {ckpt_path} (val_acc={best_acc:.4f})")

    # 학습 완료 후 결과 저장
    save_json({'history': history}, os.path.join(args.output_dir, 'training_log.json'))
    loss_png, acc_png = plot_training_curves(history, args.output_dir)
    save_html_report(args.output_dir, best_acc, best_state['val_metrics'] if best_state else {})

    # 테스트셋이 있으면 최종 평가 수행
    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        print("[테스트셋 평가]")
        if best_state is not None:
            model.load_state_dict(best_state['model_state'])
        test_metrics = evaluate(model, test_loader, criterion, device, num_classes=len(classes))
        print_metrics(test_metrics, classes)


# =====================
# 평가/지표 출력
# =====================

def print_metrics(metrics: Dict[str, object], classes: List[str]):
    print(f"loss={metrics['loss']:.4f}, acc={metrics['acc']:.4f}")
    cm = metrics['cm']
    print("Confusion Matrix (rows=true, cols=pred):")
    header = 'pred\true	' + '	'.join([f"{c[:10]:>10}" for c in classes])
    print(header)
    for i, row in enumerate(cm):
        line = f"{classes[i][:10]:>10}	" + '	'.join([f"{v:>10}" for v in row])
        print(line)
    print('Per-class metrics:')
    for i, c in enumerate(classes):
        p = metrics['per_class_precision'][i]
        r = metrics['per_class_recall'][i]
        f1 = metrics['per_class_f1'][i]
        print(f" - {c}: precision={p:.3f} recall={r:.3f} f1={f1:.3f}")


def eval_driver(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)

    classes = ckpt['classes'] if 'classes' in ckpt else load_json(os.path.join(os.path.dirname(args.checkpoint), 'labels.json'))['classes']
    num_classes = len(classes)

    _, val_set, test_set, _, _ = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    eval_set = val_set if val_set is not None else test_set
    if eval_set is None:
        eval_set, _, _, _, _ = build_datasets(args.data_dir, args.img_size, args.val_split, args.seed)

    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(num_classes=num_classes, backbone=args.backbone)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(model, eval_loader, criterion, device, num_classes=num_classes)
    print_metrics(metrics, classes)


# =====================
# 예측 (CLI)
# =====================

def load_labels_from_ckpt_or_json(checkpoint: str) -> Tuple[List[str], Dict[str, int]]:
    ckpt_dir = os.path.dirname(checkpoint)
    labels_json = os.path.join(ckpt_dir, 'labels.json')
    classes: List[str] = []
    class_to_idx: Dict[str, int] = {}
    device = torch.device('cpu')
    try:
        ckpt = torch.load(checkpoint, map_location=device)
        if 'classes' in ckpt and 'class_to_idx' in ckpt:
            classes = ckpt['classes']
            class_to_idx = ckpt['class_to_idx']
        else:
            raise KeyError('클래스 정보 없음')
    except Exception:
        meta = load_json(labels_json)
        classes = meta['classes']
        class_to_idx = meta['class_to_idx']
    return classes, class_to_idx


def predict_on_path(args):
    """
    이미지 또는 폴더에 대해 예측 수행 (CLI 모드)
    
    Args:
        args: 명령줄 인자 (체크포인트 경로, 입력 경로 등)
    """
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # 체크포인트에서 클래스 정보 로드
    classes, _ = load_labels_from_ckpt_or_json(args.checkpoint)
    num_classes = len(classes)

    # 모델 생성 및 체크포인트 로드
    model = build_model(num_classes=num_classes, backbone=args.backbone)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
    model.to(device)
    model.eval()  # 평가 모드로 설정

    # 이미지 전처리 변환 준비
    _, eval_tf = build_transforms(img_size=args.img_size)

    # 예측할 이미지 경로 수집
    paths: List[str] = []
    if os.path.isdir(args.input):
        # 폴더인 경우: 하위 모든 이미지 파일 수집
        for root, _, files in os.walk(args.input):
            for f in files:
                if is_image_file(f):
                    paths.append(os.path.join(root, f))
    else:
        # 단일 파일인 경우
        paths.append(args.input)

    # 예측 결과 저장 리스트
    results: List[Dict[str, object]] = []
    
    # 그래디언트 계산 비활성화 (예측 시 불필요)
    with torch.no_grad():
        for p in sorted(paths):
            try:
                # 이미지 로드 및 전처리
                img = Image.open(p).convert('RGB')
                tensor = eval_tf(img).unsqueeze(0).to(device)  # 배치 차원 추가 [1, C, H, W]
                
                # 모델 예측
                logits = model(tensor)
                
                # 소프트맥스를 통해 확률로 변환
                probs = torch.softmax(logits, dim=1).squeeze(0)
                
                # 가장 높은 확률의 클래스 선택
                conf, pred_idx = torch.max(probs, dim=0)
                pred_label = classes[pred_idx.item()]
                
                # 결과 저장 (모든 클래스별 확률 포함)
                result = {
                    'path': p,
                    'pred_label': pred_label,
                    'confidence': float(conf.item()),
                    'probs': {classes[i]: float(probs[i].item()) for i in range(len(classes))}
                }
                results.append(result)
            except Exception as e:
                # 오류 발생 시 오류 정보 저장
                results.append({'path': p, 'error': str(e)})

    # 예측 결과 출력
    print("[예측 결과]")
    for r in results:
        if 'error' in r:
            print(f"{r['path']} -> ERROR: {r['error']}")
        else:
            print(f"{r['path']} -> {r['pred_label']} (conf={r['confidence']:.3f})")
            
            # --show_probs 옵션이 활성화된 경우 모든 클래스별 확률 출력
            if args.show_probs:
                # 확률을 높은 순으로 정렬해서 모두 표시
                sorted_probs = sorted(r['probs'].items(), key=lambda x: x[1], reverse=True)
                for cls, pr in sorted_probs:
                    print(f"   - {cls}: {pr:.4f}")

    # 결과를 JSON 파일로 저장
    if args.output_dir:
        ensure_dir(args.output_dir)
        out_path = os.path.join(args.output_dir, 'predictions.json')
        save_json({'results': results}, out_path)
        print(f"저장: {out_path}")


# =====================
# GUI (tkinter)
# =====================
class GalaxyGUI:
    def __init__(self, checkpoint: Optional[str] = None, backbone: str = 'resnet18', img_size: int = 224, cpu: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        self.backbone = backbone
        self.img_size = img_size
        self.cpu = cpu
        
        # 모델 관련 변수
        self.model = None
        self.classes = None
        self.num_classes = 0
        self.eval_tf = None
        
        # 필수 폴더 자동 생성
        self._ensure_initial_folders()
        
        # 체크포인트가 제공된 경우 모델 로드
        if checkpoint and os.path.isfile(checkpoint):
            self._load_checkpoint(checkpoint)
        
        # 학습 관련 변수
        self.training_thread = None
        self.is_training = False
        self.should_stop_training = False
        self.train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Tk 초기화 (플롯 임베드를 위해 백엔드를 다시 TkAgg로 설정)
        matplotlib.use('TkAgg')
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.FigureCanvasTkAgg = FigureCanvasTkAgg

        self.root = tk.Tk()
        self.root.title('은하 분류기 - 통합 GUI')
        self._build_ui()

        self.current_image_path: Optional[str] = None
        self.current_image_pil: Optional[Image.Image] = None
    
    def _ensure_initial_folders(self):
        """프로그램 시작 시 기본 폴더 구조 생성"""
        # outputs 폴더는 항상 생성
        ensure_dir('./outputs')
        
        # dataset 폴더가 없으면 생성 (단, train 폴더는 사용자가 데이터를 넣을 때까지 대기)
        if not os.path.exists('./dataset'):
            ensure_dir('./dataset')
            print("[안내] dataset 폴더를 생성했습니다.")
            print("       학습을 시작하려면 다음 구조로 데이터를 준비하세요:")
            print("       ./dataset/train/클래스명/이미지파일들")
    
    def _load_checkpoint(self, checkpoint: str):
        """체크포인트에서 모델 로드"""
        try:
            self.classes, _ = load_labels_from_ckpt_or_json(checkpoint)
            self.num_classes = len(self.classes)
            self.model = build_model(num_classes=self.num_classes, backbone=self.backbone)
            ckpt = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
            self.model.to(self.device)
            self.model.eval()
            _, self.eval_tf = build_transforms(self.img_size)
            print(f"[성공] 모델을 로드했습니다: {checkpoint}")
        except Exception as e:
            print(f"[경고] 체크포인트 로드 실패: {e}")

    def _build_ui(self):
        self.root.geometry('1200x700')
        self.root.minsize(1000, 600)

        # 탭 컨트롤 생성
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 학습 탭
        self.train_tab = ttk.Frame(notebook)
        notebook.add(self.train_tab, text='학습')
        self._build_train_tab()
        
        # 예측 탭
        self.predict_tab = ttk.Frame(notebook)
        notebook.add(self.predict_tab, text='예측')
        self._build_predict_tab()
        
        # 상태바
        status = ttk.Frame(self.root)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value='준비 완료')
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, padx=8, pady=4)
    
    def _build_train_tab(self):
        """학습 탭 UI 구성"""
        # 상단: 설정 프레임
        settings_frame = ttk.LabelFrame(self.train_tab, text='학습 설정')
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 데이터셋 경로
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(row1, text='데이터셋 경로:', width=15).pack(side=tk.LEFT)
        self.train_data_dir = tk.StringVar(value='./dataset')
        ttk.Entry(row1, textvariable=self.train_data_dir, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text='찾아보기', command=self._browse_dataset).pack(side=tk.LEFT)
        
        # 하이퍼파라미터
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(row2, text='Epochs:', width=10).pack(side=tk.LEFT)
        self.train_epochs = tk.IntVar(value=30)
        ttk.Spinbox(row2, from_=1, to=200, textvariable=self.train_epochs, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row2, text='Batch Size:', width=10).pack(side=tk.LEFT, padx=10)
        self.train_batch_size = tk.IntVar(value=32)
        ttk.Spinbox(row2, from_=4, to=128, textvariable=self.train_batch_size, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row2, text='Learning Rate:', width=12).pack(side=tk.LEFT, padx=10)
        self.train_lr = tk.StringVar(value='0.0003')
        ttk.Entry(row2, textvariable=self.train_lr, width=10).pack(side=tk.LEFT, padx=5)
        
        row3 = ttk.Frame(settings_frame)
        row3.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(row3, text='백본:', width=10).pack(side=tk.LEFT)
        self.train_backbone = tk.StringVar(value='resnet18')
        ttk.Combobox(row3, textvariable=self.train_backbone, values=['resnet18', 'resnet34'], width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row3, text='이미지 크기:', width=12).pack(side=tk.LEFT, padx=10)
        self.train_img_size = tk.IntVar(value=224)
        ttk.Spinbox(row3, from_=128, to=512, increment=32, textvariable=self.train_img_size, width=10).pack(side=tk.LEFT, padx=5)
        
        self.train_use_amp = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text='혼합 정밀도 (AMP)', variable=self.train_use_amp).pack(side=tk.LEFT, padx=20)
        
        # 출력 경로
        row4 = ttk.Frame(settings_frame)
        row4.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(row4, text='출력 경로:', width=15).pack(side=tk.LEFT)
        self.train_output_dir = tk.StringVar(value='./outputs')
        ttk.Entry(row4, textvariable=self.train_output_dir, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(row4, text='찾아보기', command=self._browse_output).pack(side=tk.LEFT)
        
        # 컨트롤 버튼
        btn_frame = ttk.Frame(settings_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.train_btn = ttk.Button(btn_frame, text='학습 시작', command=self._start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text='중지', command=self._stop_training, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='체크포인트 로드', command=self._load_checkpoint_dialog).pack(side=tk.LEFT, padx=5)
        
        # 하단: 진행 상황 프레임
        progress_frame = ttk.LabelFrame(self.train_tab, text='학습 진행 상황')
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 왼쪽: 로그
        left_frame = ttk.Frame(progress_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(left_frame, text='학습 로그:').pack(anchor=tk.W)
        self.train_log = tk.Text(left_frame, height=15, width=50)
        self.train_log.pack(fill=tk.BOTH, expand=True)
        log_scroll = ttk.Scrollbar(left_frame, command=self.train_log.yview)
        self.train_log.config(yscrollcommand=log_scroll.set)
        
        # 오른쪽: 차트
        right_frame = ttk.Frame(progress_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.train_fig, (self.train_ax1, self.train_ax2) = plt.subplots(2, 1, figsize=(6, 6), dpi=80)
        self.train_canvas = self.FigureCanvasTkAgg(self.train_fig, master=right_frame)
        self.train_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.train_ax1.set_title('Loss')
        self.train_ax1.set_xlabel('Epoch')
        self.train_ax1.set_ylabel('Loss')
        self.train_ax1.legend(['Train', 'Val'])
        self.train_ax1.grid(True)
        
        self.train_ax2.set_title('Accuracy')
        self.train_ax2.set_xlabel('Epoch')
        self.train_ax2.set_ylabel('Accuracy')
        self.train_ax2.legend(['Train', 'Val'])
        self.train_ax2.grid(True)
        
        self.train_fig.tight_layout()
    
    def _build_predict_tab(self):
        """예측 탭 UI 구성"""
        # 상단 컨트롤
        top = ttk.Frame(self.predict_tab)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        ttk.Button(top, text='이미지 열기', command=self.on_open_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text='폴더 열기', command=self.on_open_folder).pack(side=tk.LEFT, padx=4)
        self.path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.path_var, width=50).pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        ttk.Button(top, text='예측', command=self.on_predict).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text='체크포인트 로드', command=self._load_checkpoint_dialog).pack(side=tk.LEFT, padx=4)

        # 메인 영역: 좌(이미지) - 우(표+바차트)
        main = ttk.Frame(self.predict_tab)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 좌 - 이미지 캔버스
        left = ttk.LabelFrame(main, text='이미지')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.img_label = ttk.Label(left)
        self.img_label.pack(fill=tk.BOTH, expand=True)

        # 우 - 표 + 차트 + 피드백
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        table_frame = ttk.LabelFrame(right, text='클래스별 확률')
        table_frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(table_frame, columns=('class','prob'), show='headings', height=8)
        self.tree.heading('class', text='Class')
        self.tree.heading('prob', text='Probability')
        self.tree.column('class', width=160)
        self.tree.column('prob', width=120)
        self.tree.pack(fill=tk.BOTH, expand=True)

        chart_frame = ttk.LabelFrame(right, text='확률 바 차트')
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=8)
        self.fig, self.ax = plt.subplots(figsize=(5,2.5), dpi=100)
        self.canvas = self.FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 피드백 프레임 (예측 강화 학습)
        feedback_frame = ttk.LabelFrame(right, text='예측 피드백 (강화 학습)')
        feedback_frame.pack(fill=tk.X, pady=8)
        
        feedback_info = ttk.Frame(feedback_frame)
        feedback_info.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(feedback_info, text='예측이 맞나요?', font=('', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        btn_frame = ttk.Frame(feedback_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.feedback_correct_btn = ttk.Button(btn_frame, text='✓ 정답', command=self._on_correct_prediction, state='disabled')
        self.feedback_correct_btn.pack(side=tk.LEFT, padx=2)
        
        self.feedback_wrong_btn = ttk.Button(btn_frame, text='✗ 오답 (수정)', command=self._on_wrong_prediction, state='disabled')
        self.feedback_wrong_btn.pack(side=tk.LEFT, padx=2)
        
        # 데이터셋 추가 정보
        info_label = ttk.Label(feedback_frame, 
                               text='정답: 학습 데이터에 추가 | 오답: 올바른 클래스 선택 후 추가',
                               font=('', 8), foreground='gray')
        info_label.pack(padx=5, pady=2)
        
        # 현재 예측 결과 저장용
        self.current_prediction_result = None

    
    def _browse_dataset(self):
        path = filedialog.askdirectory(title='데이터셋 폴더 선택')
        if path:
            self.train_data_dir.set(path)
    
    def _browse_output(self):
        path = filedialog.askdirectory(title='출력 폴더 선택')
        if path:
            self.train_output_dir.set(path)
    
    def _load_checkpoint_dialog(self):
        path = filedialog.askopenfilename(
            title='체크포인트 파일 선택',
            filetypes=[('PyTorch 모델', '*.pt *.pth'), ('모든 파일', '*.*')]
        )
        if path:
            self._load_checkpoint(path)
            messagebox.showinfo('성공', '모델을 로드했습니다!')
            self.status_var.set(f'모델 로드 완료: {os.path.basename(path)}')
    
    def _start_training(self):
        """학습 시작"""
        if self.is_training:
            messagebox.showwarning('경고', '이미 학습이 진행 중입니다.')
            return
        
        # 입력 검증
        data_dir = self.train_data_dir.get()
        if not data_dir:
            messagebox.showerror('오류', '데이터셋 경로를 입력해주세요.')
            return
        
        # 데이터셋 루트 폴더가 없으면 생성
        if not os.path.isdir(data_dir):
            try:
                ensure_dir(data_dir)
                messagebox.showinfo('정보', 
                    f'데이터셋 폴더를 생성했습니다: {data_dir}\n\n'
                    f'다음 구조로 데이터를 준비해주세요:\n'
                    f'{data_dir}/train/클래스명/이미지파일들')
                return
            except Exception as e:
                messagebox.showerror('오류', f'폴더 생성 실패: {str(e)}')
                return
        
        train_dir = os.path.join(data_dir, 'train')
        if not os.path.isdir(train_dir):
            # train 폴더 생성 및 안내
            try:
                ensure_dir(train_dir)
                messagebox.showinfo('정보',
                    f'train 폴더를 생성했습니다: {train_dir}\n\n'
                    f'다음 구조로 데이터를 준비해주세요:\n'
                    f'{train_dir}/클래스명/이미지파일들\n\n'
                    f'예시:\n'
                    f'{train_dir}/Spiral/image1.jpg\n'
                    f'{train_dir}/Elliptical/image2.jpg\n'
                    f'{train_dir}/Irregular/image3.jpg')
                return
            except Exception as e:
                messagebox.showerror('오류', f'train 폴더 생성 실패: {str(e)}')
                return
        
        # train 폴더 내에 클래스 폴더가 있는지 확인
        class_dirs = [d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')]
        if not class_dirs:
            messagebox.showerror('오류',
                f'train 폴더에 클래스 폴더가 없습니다.\n\n'
                f'다음과 같이 클래스별 폴더를 만들고 이미지를 넣어주세요:\n'
                f'{train_dir}/클래스명1/\n'
                f'{train_dir}/클래스명2/\n\n'
                f'예시:\n'
                f'{train_dir}/Spiral/\n'
                f'{train_dir}/Elliptical/\n'
                f'{train_dir}/Irregular/')
            return
        
        # 각 클래스 폴더에 이미지가 있는지 확인
        empty_classes = []
        for cls in class_dirs:
            cls_path = os.path.join(train_dir, cls)
            images = [f for f in os.listdir(cls_path) if is_image_file(f)]
            if not images:
                empty_classes.append(cls)
        
        if empty_classes:
            messagebox.showwarning('경고',
                f'다음 클래스 폴더에 이미지가 없습니다:\n' + '\n'.join(empty_classes) +
                f'\n\n이미지를 추가한 후 다시 시도하세요.')
            return
        
        # 학습 파라미터 수집
        try:
            lr = float(self.train_lr.get())
        except:
            messagebox.showerror('오류', 'Learning Rate는 숫자여야 합니다.')
            return
        
        # 출력 폴더 자동 생성
        output_dir = self.train_output_dir.get()
        if output_dir:
            ensure_dir(output_dir)
        
        # UI 상태 변경
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.is_training = True
        self.should_stop_training = False
        self.train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.train_log.delete('1.0', tk.END)
        
        # 별도 스레드에서 학습 실행
        import threading
        self.training_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.training_thread.start()
    
    def _stop_training(self):
        """학습 중지"""
        self.should_stop_training = True
        self.status_var.set('학습 중지 요청됨...')
    
    def _train_worker(self):
        """
        학습 워커 (별도 스레드에서 실행)
        GUI의 메인 스레드를 블록하지 않으면서 학습을 진행
        """
        try:
            # 재현성을 위한 시드 설정
            set_seed(42)
            device = self.device
            
            # 데이터셋 로드 및 전처리
            self._log_message('데이터셋 로딩 중...\n')
            train_set, val_set, test_set, classes, class_to_idx = build_datasets(
                data_dir=self.train_data_dir.get(),
                img_size=self.train_img_size.get(),
                val_split=0.15,
                seed=42,
            )
            self._log_message(f'클래스: {classes}\n')
            self._log_message(f'학습 샘플 수: {len(train_set)}, 검증 샘플 수: {len(val_set)}\n\n')
            
            # 데이터 로더 생성
            batch_size = self.train_batch_size.get()
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
            
            # 모델 구축 및 GPU/CPU로 이동
            self._log_message('모델 구축 중...\n')
            model = build_model(num_classes=len(classes), backbone=self.train_backbone.get())
            model.to(device)
            
            # 손실 함수, 옵티마이저, 스케줄러 설정
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=float(self.train_lr.get()), weight_decay=1e-4)
            epochs = self.train_epochs.get()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda' and self.train_use_amp.get()) else None
            
            # 최고 성능 모델 추적
            best_acc = -1.0
            best_state = None
            output_dir = self.train_output_dir.get()
            ensure_dir(output_dir)
            
            # 클래스 레이블 정보 저장
            labels_path = os.path.join(output_dir, 'labels.json')
            save_json({'classes': classes, 'class_to_idx': class_to_idx}, labels_path)
            
            self._log_message(f'학습 시작 (총 {epochs} epoch)\n')
            self._log_message('='*50 + '\n\n')
            
            # 에포크별 학습 루프
            for epoch in range(1, epochs + 1):
                # 중지 요청 확인
                if self.should_stop_training:
                    self._log_message('\n학습이 사용자에 의해 중지되었습니다.\n')
                    break
                
                t0 = time.time()
                
                # 1 에포크 학습
                tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_clip=1.0)
                
                # 검증 데이터로 평가
                val_metrics = evaluate(model, val_loader, criterion, device, num_classes=len(classes))
                scheduler.step()
                
                # 히스토리 기록
                self.train_history['train_loss'].append(tr_loss)
                self.train_history['train_acc'].append(tr_acc)
                self.train_history['val_loss'].append(val_metrics['loss'])
                self.train_history['val_acc'].append(val_metrics['acc'])
                
                # 진행 상황 로그 출력
                elapsed = time.time() - t0
                log_msg = f"[Epoch {epoch:03d}/{epochs}] " \
                         f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | " \
                         f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} | " \
                         f"time={elapsed:.1f}s\n"
                self._log_message(log_msg)
                
                # 최고 검증 정확도 달성 시 모델 저장
                if val_metrics['acc'] > best_acc:
                    best_acc = val_metrics['acc']
                    best_state = {
                        'epoch': epoch,
                        'model_state': copy.deepcopy(model.state_dict()),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'classes': classes,
                        'class_to_idx': class_to_idx,
                        'val_metrics': val_metrics,
                    }
                    ckpt_path = os.path.join(output_dir, 'best_model.pt')
                    torch.save(best_state, ckpt_path)
                    self._log_message(f"  -> 베스트 모델 저장: {ckpt_path} (val_acc={best_acc:.4f})\n")
                
                # 실시간 차트 업데이트
                self._update_training_chart()
                
            # 학습 완료 후 처리
            if not self.should_stop_training:
                self._log_message('\n' + '='*50 + '\n')
                self._log_message(f'학습 완료! 최고 검증 정확도: {best_acc:.4f}\n')
                
                # 로그 및 리포트 저장
                save_json({'history': self.train_history}, os.path.join(output_dir, 'training_log.json'))
                plot_training_curves(self.train_history, output_dir)
                if best_state:
                    save_html_report(output_dir, best_acc, best_state['val_metrics'])
                self._log_message(f'\n결과가 {output_dir}에 저장되었습니다.\n')
                
                # 학습된 모델을 현재 세션에 로드 (예측 탭에서 바로 사용 가능)
                if best_state:
                    self.model = model
                    self.model.load_state_dict(best_state['model_state'])
                    self.model.eval()
                    self.classes = classes
                    self.num_classes = len(classes)
                    _, self.eval_tf = build_transforms(self.train_img_size.get())
                    self._log_message('학습된 모델이 예측 탭에서 사용 가능합니다.\n')
                
                # 완료 메시지 표시
                self.root.after(0, lambda: messagebox.showinfo('완료', f'학습이 완료되었습니다!\n최고 검증 정확도: {best_acc:.4f}'))
            
        except Exception as e:
            # 오류 발생 시 로그 출력
            self._log_message(f'\n오류 발생: {str(e)}\n')
            import traceback
            self._log_message(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror('오류', f'학습 중 오류가 발생했습니다:\n{str(e)}'))
        
        finally:
            # UI 상태 복원 (메인 스레드에서 실행)
            self.is_training = False
            self.root.after(0, self._training_finished)
    
    def _training_finished(self):
        """학습 종료 후 UI 정리"""
        self.train_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set('학습 완료')
    
    def _log_message(self, msg: str):
        """학습 로그에 메시지 추가"""
        def append():
            self.train_log.insert(tk.END, msg)
            self.train_log.see(tk.END)
        self.root.after(0, append)
    
    def _update_training_chart(self):
        """학습 곡선 차트 업데이트"""
        def update():
            # 한글 폰트 재설정
            setup_korean_font()
            
            # Loss 차트
            self.train_ax1.clear()
            epochs = range(1, len(self.train_history['train_loss']) + 1)
            self.train_ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='Train')
            self.train_ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Val')
            self.train_ax1.set_title('Loss')
            self.train_ax1.set_xlabel('Epoch')
            self.train_ax1.set_ylabel('Loss')
            self.train_ax1.legend()
            self.train_ax1.grid(True)
            
            # Accuracy 차트
            self.train_ax2.clear()
            self.train_ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='Train')
            self.train_ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='Val')
            self.train_ax2.set_title('Accuracy')
            self.train_ax2.set_xlabel('Epoch')
            self.train_ax2.set_ylabel('Accuracy')
            self.train_ax2.legend()
            self.train_ax2.grid(True)
            
            self.train_fig.tight_layout()
            self.train_canvas.draw()
        
        self.root.after(0, update)

    def run(self):
        self.root.mainloop()

    def on_open_image(self):
        path = filedialog.askopenfilename(filetypes=[
            ('이미지 파일', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.gif *.jfif *.ico'),
            ('모든 파일', '*.*')
        ])
        if path:
            self.path_var.set(path)
            self._load_preview(path)

    def on_open_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.path_var.set(path)
            # 폴더는 미리보기 생략

    def _load_preview(self, path: str):
        try:
            pil = Image.open(path).convert('RGB')
            self.current_image_pil = pil
            self.current_image_path = path
            # 미리보기 크기 조정
            w, h = pil.size
            maxw, maxh = 600, 500
            scale = min(maxw / w, maxh / h, 1.0)
            disp = pil.resize((int(w*scale), int(h*scale)))
            tk_img = ImageTk.PhotoImage(disp)
            self.img_label.configure(image=tk_img)
            self.img_label.image = tk_img
            self.status_var.set(f'불러옴: {path}')
        except Exception as e:
            messagebox.showerror('오류', str(e))

    def on_predict(self):
        if self.model is None or self.classes is None:
            messagebox.showerror('오류', '먼저 모델을 로드해주세요.')
            return
        
        path = self.path_var.get().strip()
        if not path:
            messagebox.showinfo('안내', '이미지 또는 폴더를 선택하세요.')
            return
        if os.path.isdir(path):
            # 폴더 예측: 첫 1개만 미리보기, 표/차트는 마지막 결과로 갱신
            imgs = []
            for root, _, files in os.walk(path):
                for f in files:
                    if is_image_file(f):
                        imgs.append(os.path.join(root, f))
            if not imgs:
                messagebox.showinfo('안내', '이미지 파일이 없습니다.')
                return
            self._load_preview(imgs[0])
            last = None
            for p in imgs:
                last = self._predict_single(p)
            if last is not None:
                self._render_result(last)
        else:
            if not os.path.isfile(path):
                messagebox.showerror('오류', '파일을 찾을 수 없습니다.')
                return
            self._load_preview(path)
            result = self._predict_single(path)
            if result is not None:
                self._render_result(result)

    def _predict_single(self, path: str) -> Optional[Dict[str, object]]:
        """
        단일 이미지에 대해 예측 수행 (GUI 모드)
        
        Args:
            path: 예측할 이미지 파일 경로
        
        Returns:
            예측 결과 딕셔너리 (오류 발생 시 None 반환)
            - path: 이미지 경로
            - pred_label: 예측된 클래스명
            - confidence: 예측 신뢰도 (최대 확률값)
            - probs: 모든 클래스별 확률 딕셔너리
        """
        try:
            # 이미지 로드 및 RGB로 변환
            img = Image.open(path).convert('RGB')
            
            # 이미지 전처리 및 배치 차원 추가
            tensor = self.eval_tf(img).unsqueeze(0).to(self.device)
            
            # 모델 예측 (그래디언트 계산 불필요)
            with torch.no_grad():
                logits = self.model(tensor)  # 로짓(raw 출력값) 계산
                probs = torch.softmax(logits, dim=1).squeeze(0)  # 소프트맥스로 확률 변환
                conf, pred_idx = torch.max(probs, dim=0)  # 최대 확률 및 해당 클래스 인덱스
            
            # 예측된 클래스명
            pred_label = self.classes[pred_idx.item()]
            
            # 결과 딕셔너리 생성
            result = {
                'path': path,
                'pred_label': pred_label,
                'confidence': float(conf.item()),
                'probs': {self.classes[i]: float(probs[i].item()) for i in range(self.num_classes)}
            }
            return result
        except Exception as e:
            # 오류 발생 시 사용자에게 알림
            messagebox.showerror('오류', str(e))
            return None

    def _render_result(self, result: Dict[str, object]):
        # 테이블 갱신
        for it in self.tree.get_children():
            self.tree.delete(it)
        sorted_probs = sorted(result['probs'].items(), key=lambda x: x[1], reverse=True)
        for cls, pr in sorted_probs:
            self.tree.insert('', tk.END, values=(cls, f"{pr:.4f}"))
        self.status_var.set(f"예측: {result['pred_label']} (conf={result['confidence']:.3f})")
        
        # 차트 갱신
        self.ax.clear()
        labels = [x[0] for x in sorted_probs]
        values = [x[1] for x in sorted_probs]
        self.ax.bar(labels, values)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel('Probability')
        self.ax.set_xlabel('Class')
        
        # 한글 폰트 재설정 (GUI 모드에서 TkAgg 백엔드 사용 시)
        try:
            if sys.platform.startswith('win'):
                self.ax.set_xticklabels(labels, rotation=30, ha='right', fontproperties=fm.FontProperties(family='Malgun Gothic'))
            elif sys.platform.startswith('darwin'):
                self.ax.set_xticklabels(labels, rotation=30, ha='right', fontproperties=fm.FontProperties(family='AppleGothic'))
            else:
                self.ax.set_xticklabels(labels, rotation=30, ha='right', fontproperties=fm.FontProperties(family='NanumGothic'))
        except:
            self.ax.set_xticklabels(labels, rotation=30, ha='right')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # 현재 예측 결과 저장 및 피드백 버튼 활성화
        self.current_prediction_result = result
        self.feedback_correct_btn.config(state='normal')
        self.feedback_wrong_btn.config(state='normal')
    
    def _on_correct_prediction(self):
        """
        예측이 맞을 때 호출: 해당 이미지를 학습 데이터에 추가
        
        강화 학습 메커니즘의 일부로, 모델이 올바르게 예측한 이미지를
        학습 데이터셋에 추가하여 향후 재학습 시 모델 성능을 강화함
        """
        if not self.current_prediction_result or not self.classes:
            return
        
        result = self.current_prediction_result
        pred_label = result['pred_label']  # 예측된 클래스
        image_path = result['path']  # 이미지 파일 경로
        
        # 데이터셋 경로 가져오기
        dataset_dir = self.train_data_dir.get()
        if not dataset_dir or not os.path.exists(dataset_dir):
            messagebox.showerror('오류', '데이터셋 경로가 설정되지 않았습니다.\n학습 탭에서 경로를 설정하세요.')
            return
        
        # train 폴더의 해당 클래스 폴더에 이미지 복사
        try:
            # 예측된 클래스 폴더 경로 생성
            train_class_dir = os.path.join(dataset_dir, 'train', pred_label)
            ensure_dir(train_class_dir)
            
            # 파일명 중복 방지 (같은 이름이 있으면 _1, _2 등 추가)
            filename = os.path.basename(image_path)
            base, ext = os.path.splitext(filename)
            counter = 1
            dest_path = os.path.join(train_class_dir, filename)
            while os.path.exists(dest_path):
                dest_path = os.path.join(train_class_dir, f"{base}_{counter}{ext}")
                counter += 1
            
            # 이미지 복사 (메타데이터 보존)
            import shutil
            shutil.copy2(image_path, dest_path)
            
            messagebox.showinfo('성공', 
                f'이미지를 학습 데이터에 추가했습니다!\n\n'
                f'클래스: {pred_label}\n'
                f'경로: {dest_path}\n\n'
                f'재학습을 통해 모델을 개선할 수 있습니다.')
            
            self.status_var.set(f'✓ 정답 확인: {pred_label} - 데이터 추가 완료')
            
        except Exception as e:
            messagebox.showerror('오류', f'이미지 추가 실패:\n{str(e)}')
    
    def _on_wrong_prediction(self):
        """
        예측이 틀렸을 때 호출: 올바른 클래스를 선택하여 데이터 추가
        
        강화 학습 메커니즘의 핵심으로, 모델이 잘못 예측한 이미지를
        사용자가 올바른 클래스로 수정하고 학습 데이터셋에 추가하여
        향후 재학습 시 해당 오류를 개선할 수 있도록 함
        """
        if not self.current_prediction_result or not self.classes:
            return
        
        result = self.current_prediction_result
        image_path = result['path']
        pred_label = result['pred_label']
        
        # 올바른 클래스 선택 대화상자 생성
        dialog = tk.Toplevel(self.root)
        dialog.title('올바른 클래스 선택')
        dialog.geometry('400x300')
        dialog.transient(self.root)  # 부모 창에 종속
        dialog.grab_set()  # 모달 다이얼로그로 설정
        
        ttk.Label(dialog, text=f'예측 결과: {pred_label}', font=('', 11, 'bold')).pack(pady=10)
        ttk.Label(dialog, text='올바른 클래스를 선택하세요:', font=('', 10)).pack(pady=5)
        
        # 클래스 선택 리스트 박스
        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, font=('', 10))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # 모든 클래스를 리스트에 추가
        for cls in self.classes:
            listbox.insert(tk.END, cls)
        
        # 예측된 클래스를 기본 선택 (사용자 편의)
        try:
            idx = self.classes.index(pred_label)
            listbox.selection_set(idx)
            listbox.see(idx)
        except:
            pass
        
        # 사용자 선택 저장 변수
        selected_class = [None]
        
        def on_select():
            """확인 버튼 클릭 시: 선택된 클래스를 저장하고 창 닫기"""
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning('경고', '클래스를 선택하세요.')
                return
            selected_class[0] = listbox.get(selection[0])
            dialog.destroy()
        
        def on_cancel():
            """취소 버튼 클릭 시: 창만 닫기"""
            dialog.destroy()
        
        # 버튼 프레임
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text='확인', command=on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='취소', command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # 더블클릭으로도 선택 가능 (사용자 편의)
        listbox.bind('<Double-Button-1>', lambda e: on_select())
        
        # 대화상자가 닫힐 때까지 대기
        dialog.wait_window()
        
        # 선택된 클래스로 데이터 추가
        if selected_class[0]:
            self._add_to_dataset(image_path, selected_class[0], pred_label)
    
    def _add_to_dataset(self, image_path: str, correct_label: str, predicted_label: str):
        """
        이미지를 올바른 클래스의 학습 데이터에 추가
        
        Args:
            image_path: 추가할 이미지 파일 경로
            correct_label: 올바른 클래스명 (사용자가 수정한 정답)
            predicted_label: 모델이 예측한 클래스명 (틀린 예측)
        
        이 함수는 강화 학습의 핵심으로, 잘못 분류된 이미지를
        올바른 클래스 폴더에 추가하여 모델이 해당 오류를 학습할 수 있게 함
        """
        dataset_dir = self.train_data_dir.get()
        if not dataset_dir or not os.path.exists(dataset_dir):
            messagebox.showerror('오류', '데이터셋 경로가 설정되지 않았습니다.')
            return
        
        try:
            # 올바른 클래스의 train 폴더 경로
            train_class_dir = os.path.join(dataset_dir, 'train', correct_label)
            ensure_dir(train_class_dir)
            
            # 파일명 중복 방지 (이미 같은 이름이 있으면 번호 추가)
            filename = os.path.basename(image_path)
            base, ext = os.path.splitext(filename)
            counter = 1
            dest_path = os.path.join(train_class_dir, filename)
            while os.path.exists(dest_path):
                dest_path = os.path.join(train_class_dir, f"{base}_{counter}{ext}")
                counter += 1
            
            # 이미지 복사 (메타데이터 보존)
            import shutil
            shutil.copy2(image_path, dest_path)
            
            messagebox.showinfo('성공', 
                f'이미지를 올바른 클래스로 추가했습니다!\n\n'
                f'예측: {predicted_label} → 실제: {correct_label}\n'
                f'경로: {dest_path}\n\n'
                f'재학습을 통해 모델 정확도를 향상시킬 수 있습니다.')
            
            self.status_var.set(f'✗ 오답 수정: {predicted_label}→{correct_label} - 데이터 추가 완료')
            
            # 피드백 로그에 기록 (재학습 시 참고용)
            self._log_feedback(image_path, predicted_label, correct_label, dest_path)
            
        except Exception as e:
            messagebox.showerror('오류', f'이미지 추가 실패:\n{str(e)}')
    
    def _log_feedback(self, image_path: str, predicted: str, actual: str, dest_path: str):
        """
        피드백 로그 저장 (재학습 시 참고용)
        
        Args:
            image_path: 원본 이미지 경로
            predicted: 모델이 예측한 클래스
            actual: 실제(올바른) 클래스
            dest_path: 데이터셋에 추가된 이미지 경로
        
        이 함수는 모든 사용자 피드백을 JSON 파일로 기록하여
        나중에 어떤 종류의 오류가 많았는지 분석하고
        모델 개선 방향을 파악하는 데 도움을 줌
        """
        try:
            output_dir = self.train_output_dir.get()
            if not output_dir:
                output_dir = './outputs'
            ensure_dir(output_dir)
            
            log_path = os.path.join(output_dir, 'feedback_log.json')
            
            # 기존 로그 읽기 (파일이 있으면)
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # 새 피드백 항목 추가
            logs.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),  # 피드백 시간
                'original_path': image_path,  # 원본 이미지 경로
                'predicted_label': predicted,  # 잘못된 예측
                'actual_label': actual,  # 올바른 정답
                'added_to': dest_path  # 추가된 위치
            })
            
            # 업데이트된 로그를 파일로 저장
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            # 로그 저장 실패는 치명적이지 않으므로 경고만 출력
            print(f"피드백 로그 저장 실패: {e}")


# =====================
# 메인 & 인자 파서
# =====================

def build_argparser():
    p = argparse.ArgumentParser(description='은하 분류 사진 인식 프로그램 (PyTorch + GUI)')
    p.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'predict', 'gui'], help='실행 모드')
    p.add_argument('--data_dir', type=str, default='./dataset', help='데이터 루트 디렉터리 (ImageFolder 규칙)')
    p.add_argument('--input', type=str, default='', help='predict 모드에서 단일 이미지 경로 또는 폴더 경로')
    p.add_argument('--checkpoint', type=str, default='outputs/best_model.pt', help='평가/예측/GUI에 사용할 체크포인트 경로')
    p.add_argument('--output_dir', type=str, default='outputs', help='출력 디렉터리')

    # 하이퍼파라미터
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--val_split', type=float, default=0.15)
    p.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action='store_true', help='CUDA AMP 혼합정밀도 사용')
    p.add_argument('--grad_clip', type=float, default=1.0, help='그래디언트 클리핑 최대 노름(미사용은 음수)')
    p.add_argument('--cpu', action='store_true', help='강제로 CPU 사용')

    # 예측 옵션
    p.add_argument('--show_probs', action='store_true', help='CLI 예측 시 클래스별 확률 상세 출력')

    return p


def main():
    args = build_argparser().parse_args()
    
    # 출력 디렉터리 자동 생성
    ensure_dir(args.output_dir)
    
    # 학습/평가 모드일 경우 데이터 디렉터리 기본 구조 확인 및 생성
    if args.mode in ['train', 'eval']:
        data_dir = args.data_dir
        if data_dir and data_dir != './dataset':  # 기본값이 아닌 경우
            ensure_dir(data_dir)
            train_dir = os.path.join(data_dir, 'train')
            if not os.path.exists(train_dir):
                ensure_dir(train_dir)
                print(f"[안내] train 폴더를 생성했습니다: {train_dir}")
                print(f"       다음 구조로 데이터를 준비하세요:")
                print(f"       {train_dir}/클래스명1/이미지파일들")
                print(f"       {train_dir}/클래스명2/이미지파일들")
                print()

    if args.mode == 'train':
        train_driver(args)
    elif args.mode == 'eval':
        if not os.path.isdir(args.data_dir):
            print('--data_dir 가 필요합니다 (평가셋 로딩)', file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(args.checkpoint):
            print('--checkpoint 파일을 찾을 수 없습니다', file=sys.stderr)
            sys.exit(1)
        eval_driver(args)
    elif args.mode == 'predict':
        if not os.path.isfile(args.checkpoint):
            print('--checkpoint 파일을 찾을 수 없습니다', file=sys.stderr)
            sys.exit(1)
        if not args.input:
            print('--input 경로(이미지 파일 또는 폴더)를 지정하세요', file=sys.stderr)
            sys.exit(1)
        predict_on_path(args)
    elif args.mode == 'gui':
        checkpoint = args.checkpoint if os.path.isfile(args.checkpoint) else None
        if checkpoint:
            print(f'체크포인트 로드: {checkpoint}')
        else:
            print('체크포인트 없이 GUI 시작 (학습 후 예측 가능)')
        app = GalaxyGUI(checkpoint=checkpoint, backbone=args.backbone, img_size=args.img_size, cpu=args.cpu)
        app.run()


if __name__ == '__main__':
    main()
