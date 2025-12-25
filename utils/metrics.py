"""
Evaluation Metrics for Change Detection and Captioning

Supports:
- Binary CD: Precision, Recall, F1, IoU, OA, Kappa (LEVIR-MCI)
- Multi-class CD: mIoU, F1, OA (LEVIR-MCI 3-class)
- Semantic CD: SeK, mIoU, F1, OA (SECOND-CC)
- Captioning: BLEU, METEOR, ROUGE-L, CIDEr
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from collections import Counter
import math


class BinaryChangeMetrics:
    """
    Binary Change Detection Metrics for LEVIR-MCI.
    
    Computes: Precision, Recall, F1, IoU, OA, Kappa
    """
    
    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset counters."""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with batch predictions."""
        preds = predictions.cpu().numpy().flatten()
        tgts = targets.cpu().numpy().flatten()
        
        # Filter valid pixels
        mask = tgts != self.ignore_index
        preds = (preds[mask] > 0).astype(int)
        tgts = (tgts[mask] > 0).astype(int)
        
        self.tp += np.sum((preds == 1) & (tgts == 1))
        self.fp += np.sum((preds == 1) & (tgts == 0))
        self.tn += np.sum((preds == 0) & (tgts == 0))
        self.fn += np.sum((preds == 0) & (tgts == 1))
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        eps = 1e-8
        
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        
        total = self.tp + self.fp + self.tn + self.fn
        oa = (self.tp + self.tn) / (total + eps)
        
        # Kappa
        p_o = oa
        p_e = ((self.tp + self.fn) * (self.tp + self.fp) + 
               (self.tn + self.fp) * (self.tn + self.fn)) / (total ** 2 + eps)
        kappa = (p_o - p_e) / (1 - p_e + eps)
        
        return {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'IoU': iou,
            'OA': oa,
            'Kappa': kappa
        }


class SemanticChangeMetrics:
    """
    Semantic Change Detection Metrics for SECOND-CC.
    
    Computes: SeK (Separated Kappa), mIoU, F1, OA
    """
    
    def __init__(self, num_classes: int = 7, ignore_index: int = 255):
        self.num_classes = num_classes
        self.num_transitions = num_classes ** 2
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset confusion matrix."""
        self.confusion = np.zeros((self.num_transitions, self.num_transitions), dtype=np.int64)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update confusion matrix."""
        preds = predictions.cpu().numpy().flatten()
        tgts = targets.cpu().numpy().flatten()
        
        mask = tgts != self.ignore_index
        preds = np.clip(preds[mask], 0, self.num_transitions - 1)
        tgts = np.clip(tgts[mask], 0, self.num_transitions - 1)
        
        for p, t in zip(preds, tgts):
            self.confusion[int(t), int(p)] += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        cm = self.confusion
        total = cm.sum()
        eps = 1e-8
        
        if total == 0:
            return {'SeK': 0.0, 'mIoU': 0.0, 'F1': 0.0, 'OA': 0.0}
        
        # Overall Accuracy
        oa = np.diag(cm).sum() / total
        
        # Separated Kappa
        p_o = oa
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        p_e = (row_sums * col_sums).sum() / (total ** 2)
        sek = (p_o - p_e) / (1 - p_e + eps)
        
        # mIoU
        intersection = np.diag(cm)
        union = row_sums + col_sums - intersection
        valid = union > 0
        iou = np.zeros(self.num_transitions)
        iou[valid] = intersection[valid] / union[valid]
        miou = iou[valid].mean() if valid.any() else 0.0
        
        # F1 (macro)
        tp = np.diag(cm)
        fp = col_sums - tp
        fn = row_sums - tp
        
        precision = np.zeros(self.num_transitions)
        recall = np.zeros(self.num_transitions)
        
        valid_p = (tp + fp) > 0
        valid_r = (tp + fn) > 0
        
        precision[valid_p] = tp[valid_p] / (tp + fp)[valid_p]
        recall[valid_r] = tp[valid_r] / (tp + fn)[valid_r]
        
        f1 = np.zeros(self.num_transitions)
        valid_f1 = (precision + recall) > 0
        f1[valid_f1] = 2 * precision[valid_f1] * recall[valid_f1] / (precision + recall)[valid_f1]
        
        return {
            'SeK': float(sek),
            'mIoU': float(miou),
            'F1': float(f1.mean()),
            'OA': float(oa)
        }


class MultiClassChangeMetrics:
    """
    Multi-class Change Detection Metrics.

    Computes: mIoU, F1 (macro), OA.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset confusion matrix."""
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update confusion matrix."""
        preds = predictions.cpu().numpy().flatten()
        tgts = targets.cpu().numpy().flatten()

        mask = tgts != self.ignore_index
        preds = np.clip(preds[mask], 0, self.num_classes - 1)
        tgts = np.clip(tgts[mask], 0, self.num_classes - 1)

        for p, t in zip(preds, tgts):
            self.confusion[int(t), int(p)] += 1

    def compute(self) -> Dict[str, float]:
        """Compute metrics."""
        cm = self.confusion
        total = cm.sum()
        eps = 1e-8

        if total == 0:
            return {'mIoU': 0.0, 'F1': 0.0, 'OA': 0.0}

        # Overall Accuracy
        oa = np.diag(cm).sum() / (total + eps)

        # Per-class IoU
        intersection = np.diag(cm)
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        union = row_sums + col_sums - intersection
        valid = union > 0
        iou = np.zeros(self.num_classes)
        iou[valid] = intersection[valid] / (union[valid] + eps)
        miou = iou[valid].mean() if valid.any() else 0.0

        # Per-class F1
        tp = np.diag(cm)
        fp = col_sums - tp
        fn = row_sums - tp
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        valid_p = (tp + fp) > 0
        valid_r = (tp + fn) > 0
        precision[valid_p] = tp[valid_p] / (tp + fp)[valid_p]
        recall[valid_r] = tp[valid_r] / (tp + fn)[valid_r]
        f1 = np.zeros(self.num_classes)
        valid_f1 = (precision + recall) > 0
        f1[valid_f1] = 2 * precision[valid_f1] * recall[valid_f1] / (precision + recall)[valid_f1]

        return {
            'mIoU': float(miou),
            'F1': float(f1.mean()),
            'OA': float(oa)
        }


class CaptionMetrics:
    """
    Caption Generation Metrics.
    
    Computes: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset stored predictions and references."""
        self.predictions = []
        self.references = []
    
    def update(self, predictions: List[str], references: List[List[str]]):
        """
        Update with batch predictions.
        
        Args:
            predictions: List of predicted caption strings
            references: List of reference caption lists (multiple refs per sample)
        """
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().strip().split()
    
    def _ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-gram counts."""
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    def _bleu_score(self, pred: str, refs: List[str], n: int) -> float:
        """Compute BLEU-n for single sample."""
        pred_tokens = self._tokenize(pred)
        
        if len(pred_tokens) < n:
            return 0.0
        
        pred_ngrams = self._ngrams(pred_tokens, n)
        
        # Clipped counts from all references
        max_ref_counts = Counter()
        for ref in refs:
            ref_tokens = self._tokenize(ref)
            ref_ngrams = self._ngrams(ref_tokens, n)
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
        
        # Clipped precision
        clipped = sum(min(count, max_ref_counts[ngram]) 
                     for ngram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        
        if total == 0:
            return 0.0
        
        return clipped / total
    
    def _brevity_penalty(self, pred: str, refs: List[str]) -> float:
        """Compute brevity penalty."""
        pred_len = len(self._tokenize(pred))
        
        # Find closest reference length
        ref_lens = [len(self._tokenize(ref)) for ref in refs]
        closest = min(ref_lens, key=lambda x: (abs(x - pred_len), x))
        
        if pred_len >= closest:
            return 1.0
        return math.exp(1 - closest / (pred_len + 1e-8))
    
    def _compute_bleu(self, n: int) -> float:
        """Compute corpus BLEU-n."""
        scores = []
        for pred, refs in zip(self.predictions, self.references):
            if isinstance(refs, str):
                refs = [refs]
            score = self._bleu_score(pred, refs, n)
            bp = self._brevity_penalty(pred, refs)
            scores.append(score * bp)
        return np.mean(scores) if scores else 0.0
    
    def _compute_meteor(self) -> float:
        """Simplified METEOR score."""
        scores = []
        for pred, refs in zip(self.predictions, self.references):
            if isinstance(refs, str):
                refs = [refs]
            
            pred_tokens = set(self._tokenize(pred))
            
            best_score = 0.0
            for ref in refs:
                ref_tokens = set(self._tokenize(ref))
                
                if not pred_tokens or not ref_tokens:
                    continue
                
                matches = len(pred_tokens & ref_tokens)
                precision = matches / len(pred_tokens)
                recall = matches / len(ref_tokens)
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    best_score = max(best_score, f1)
            
            scores.append(best_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _lcs(self, s1: List[str], s2: List[str]) -> int:
        """Longest Common Subsequence length."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _compute_rouge_l(self) -> float:
        """Compute ROUGE-L F1."""
        scores = []
        for pred, refs in zip(self.predictions, self.references):
            if isinstance(refs, str):
                refs = [refs]
            
            pred_tokens = self._tokenize(pred)
            
            best_score = 0.0
            for ref in refs:
                ref_tokens = self._tokenize(ref)
                
                if not pred_tokens or not ref_tokens:
                    continue
                
                lcs = self._lcs(pred_tokens, ref_tokens)
                precision = lcs / len(pred_tokens)
                recall = lcs / len(ref_tokens)
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    best_score = max(best_score, f1)
            
            scores.append(best_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _compute_cider(self) -> float:
        """Simplified CIDEr score."""
        # Build document frequency
        doc_freq = Counter()
        all_refs = []
        for refs in self.references:
            if isinstance(refs, str):
                refs = [refs]
            for ref in refs:
                tokens = self._tokenize(ref)
                doc_freq.update(set(tokens))
                all_refs.append(tokens)
        
        num_docs = len(all_refs)
        
        scores = []
        for pred, refs in zip(self.predictions, self.references):
            if isinstance(refs, str):
                refs = [refs]
            
            pred_tokens = self._tokenize(pred)
            pred_counter = Counter(pred_tokens)
            
            # TF-IDF for prediction
            pred_tfidf = {}
            for token, count in pred_counter.items():
                tf = count / (len(pred_tokens) + 1e-8)
                idf = math.log((num_docs + 1) / (doc_freq[token] + 1))
                pred_tfidf[token] = tf * idf
            
            best_sim = 0.0
            for ref in refs:
                ref_tokens = self._tokenize(ref)
                ref_counter = Counter(ref_tokens)
                
                ref_tfidf = {}
                for token, count in ref_counter.items():
                    tf = count / (len(ref_tokens) + 1e-8)
                    idf = math.log((num_docs + 1) / (doc_freq[token] + 1))
                    ref_tfidf[token] = tf * idf
                
                # Cosine similarity
                all_tokens = set(pred_tfidf.keys()) | set(ref_tfidf.keys())
                dot = sum(pred_tfidf.get(t, 0) * ref_tfidf.get(t, 0) for t in all_tokens)
                norm1 = math.sqrt(sum(v**2 for v in pred_tfidf.values()) + 1e-8)
                norm2 = math.sqrt(sum(v**2 for v in ref_tfidf.values()) + 1e-8)
                
                sim = dot / (norm1 * norm2)
                best_sim = max(best_sim, sim)
            
            scores.append(best_sim * 10)  # Scale by 10 like original CIDEr
        
        return np.mean(scores) if scores else 0.0
    
    def compute(self) -> Dict[str, float]:
        """Compute all caption metrics."""
        return {
            'BLEU-1': self._compute_bleu(1),
            'BLEU-2': self._compute_bleu(2),
            'BLEU-3': self._compute_bleu(3),
            'BLEU-4': self._compute_bleu(4),
            'METEOR': self._compute_meteor(),
            'ROUGE-L': self._compute_rouge_l(),
            'CIDEr': self._compute_cider()
        }
