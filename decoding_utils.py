from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import mne

from fold_utils import get_sequential_folds, get_zero_shot_folds
import metrics
from plot_utils import plot_cv_results, plot_training_history


def should_update_best(current_val, best_val, smaller_is_better):
    """
    Determine if current validation value is better than best.

    Returns:
        bool: True if current value is better
    """
    if smaller_is_better:
        return current_val < best_val
    else:
        return current_val > best_val


def should_update_gradient_accumulation(
    batch_idx, total_batches, grad_accumulation_steps
):
    """
    Determine if optimizer should step based on gradient accumulation.

    Returns:
        bool: True if optimizer should step
    """
    return (batch_idx + 1) % grad_accumulation_steps == 0 or (
        batch_idx + 1
    ) == total_batches


def get_predictions(X, model, device):
    X = X.to(device)

    # Step 2: Get predicted embeddings for all neural data
    model_predictions = []
    for i in range(len(X)):
        with torch.no_grad():
            input_data = X[i : i + 1]
            pred = model(input_data)

        # Only squeeze the batch dimension (first dimension), keep the embedding dimension
        if pred.dim() > 1:
            pred = pred.squeeze(0)  # Remove only the first dimension
        model_predictions.append(pred)

    # Stack to ensure we get a 2D tensor [num_samples, embedding_dim]
    return torch.stack(model_predictions)


def build_vocabulary(words):
    """
    Build vocabulary mappings from a list of words.

    Args:
        words: List of words (may contain repetitions)

    Returns:
        word_to_id: Dictionary mapping word -> unique_id
        id_to_word: Dictionary mapping unique_id -> word
        postion_to_id: Numpy array specifying what the ith word maps to which unique id.
    """
    word_to_id = {}
    position_to_id = []
    next_id = 0

    for word in words:
        if word not in word_to_id:
            # First time seeing this word - assign new ID
            word_to_id[word] = next_id
            next_id += 1

        # Add current position to the word's position list
        word_id = word_to_id[word]
        position_to_id.append(word_id)

    # Build reverse mapping
    id_to_word = {word_id: word for word, word_id in word_to_id.items()}

    return word_to_id, id_to_word, position_to_id


def compute_cosine_distances(predictions, word_embeddings):
    """
    Compute cosine distances between predicted embeddings and word embeddings.
    Supports ensemble predictions.

    Args:
        predictions: torch tensor of shape [num_samples, embedding_dim] or
                    [num_samples, n_ensemble, embedding_dim] for ensemble predictions
        word_embeddings: torch tensor of shape [num_words, embedding_dim]

    Returns:
        scores: torch tensor of shape [num_samples, num_words] containing
                cosine distances for each word given each prediction
    """
    # Normalize word embeddings once
    word_embeddings_norm = F.normalize(word_embeddings, p=2, dim=1)

    # Handle both 2D and 3D cases by reshaping 2D to 3D with singleton dimension
    if predictions.dim() == 2:
        # Single prediction case: [num_samples, embedding_dim] -> [num_samples, 1, embedding_dim]
        predictions = predictions.unsqueeze(1)
    elif predictions.dim() != 3:
        raise ValueError(
            f"Predictions must be 2D or 3D tensor, got {predictions.dim()}D"
        )

    # Now we have: [num_samples, n_ensemble, embedding_dim]
    num_samples, n_ensemble, embedding_dim = predictions.shape

    # Reshape to treat each ensemble prediction separately
    predictions_reshaped = predictions.view(num_samples * n_ensemble, embedding_dim)

    # Normalize ensemble predictions
    predictions_norm = F.normalize(predictions_reshaped, p=2, dim=1)

    # Compute cosine similarity for all ensemble predictions
    cosine_similarities = torch.mm(predictions_norm, word_embeddings_norm.t())
    # Shape: [num_samples * n_ensemble, num_words]

    # Convert to cosine distance
    cosine_distances = 1 - cosine_similarities

    # Reshape back to separate ensemble dimension
    cosine_distances = cosine_distances.view(
        num_samples, n_ensemble, word_embeddings.shape[0]
    )

    # Average across ensemble dimension (for single predictions, n_ensemble=1, so this is a no-op)
    return cosine_distances.mean(dim=1)  # [num_samples, num_words]


def compute_class_scores(cosine_distances, word_labels=None):
    """
    Compute class scores from cosine distances by averaging over word embeddings
    belonging to the same class and applying softmax transformation.

    This implements the logic: "we computed the cosine distance between each of
    the predicted embeddings and the embeddings of all instances of each unique
    word label. The distances were averaged across unique word labels, yielding
    one score for each word label (that is, logit). We used a Softmax
    transformation on these scores (logits)."

    Args:
        cosine_distances: torch tensor of shape [num_samples, num_words] containing
                         cosine distances between predictions and word embeddings
        word_labels: Optional torch tensor of shape [num_words] containing integer class IDs
                    for each word embedding

    Returns:
        class_probabilities: torch tensor of shape [num_samples, num_classes] containing
                           softmax probabilities for each class
        class_logits: torch tensor of shape [num_samples, num_classes] containing
                     the logits (negative averaged distances) before softmax
    """
    if word_labels is not None:
        device = cosine_distances.device
        num_samples = cosine_distances.shape[0]

        # Get unique class labels and sort them for consistent ordering
        unique_classes = torch.unique(word_labels).sort()[0]
        num_classes = len(unique_classes)

        # Initialize tensors for class-averaged distances
        class_distances = torch.zeros(num_samples, num_classes, device=device)

        # For each unique class, average the distances across all word embeddings
        # belonging to that class
        for i, class_id in enumerate(unique_classes):
            # Find indices of word embeddings belonging to this class
            class_mask = word_labels == class_id
            class_indices = torch.where(class_mask)[0]

            if len(class_indices) > 0:
                # Average distances for this class across all its word embeddings
                class_distances[:, i] = cosine_distances[:, class_indices].mean(dim=1)
    else:
        class_distances = cosine_distances
        unique_classes = torch.arange(cosine_distances.shape[0])

    # Convert distances to similarities (logits)
    # Since cosine distance = 1 - cosine_similarity, we convert back:
    # logits = 1 - distance = cosine_similarity
    class_logits = 1 - class_distances

    # Apply softmax transformation to get probabilities
    class_probabilities = F.softmax(class_logits, dim=1)

    return class_probabilities, class_logits, unique_classes


def compute_word_embedding_task_metrics(
    X_test,
    Y_test,
    model,
    device,
    selected_words,
    test_index,
    train_index,
    top_k_thresholds,
    min_train_freq_auc,
    min_test_freq_auc,
):
    """
    Calculate top-k metrics and AUC-ROC for decoding from brain data.

    Args:
        X_test: Test brain data
        Y: All word embeddings across all folds.
        model: Trained model
        device: PyTorch device
        selected_words: List of selected words for vocabulary.
        test_index: Test indices for indexing into position_to_id
        train_index: Train indices for indexing into position_to_id
        top_k_thresholds: List of k values for top-k accuracy
        min_train_freq_auc: Minimum training frequency for AUC calculation
        min_test_freq_auc: Minimum test frequency for AUC calculation

    Returns:
        dict: Dictionary containing computed metrics
    """
    results = {}

    # Put model in evaluation mode
    model.eval()

    X_test, Y_test = X_test.to(device), Y_test.to(device)

    # Get predictions
    predictions = get_predictions(X_test, model, device)

    # Compute cosine distances
    distances = compute_cosine_distances(predictions, Y_test)

    # Measure performance based on each individual word occurrence
    occurence_scores, _, _ = compute_class_scores(distances)
    occurence_scores_np = occurence_scores.cpu().numpy()
    for k_val in top_k_thresholds:
        # Labels are in order of test set since we are hoping the ith example is predicted as the ith class.
        results[f"test_occ_top_{k_val}"] = metrics.top_k_accuracy(
            occurence_scores_np, np.arange(occurence_scores_np.shape[0]), k_val
        )
    results["test_occ_perplexity"] = metrics.perplexity(
        occurence_scores_np, np.arange(occurence_scores_np.shape[0])
    )

    # Group by words for an easier task.
    # Build vocabulary. While in most cases we would not want to include
    # the test set in the vocabulary building process, we remove any
    _, _, position_to_id = build_vocabulary(selected_words)
    position_to_id = np.array(position_to_id)

    word_scores, _, test_class_idxs = compute_class_scores(
        distances, torch.from_numpy(position_to_id[test_index])
    )
    # Get a mapping from over-all class index -> test class index.
    class_to_test_idxs = np.empty(np.max(position_to_id) + 1, dtype=int)
    class_to_test_idxs[test_class_idxs.cpu().numpy()] = np.arange(len(test_class_idxs))

    word_scores_np = word_scores.cpu().numpy()
    train_frequencies = np.bincount(
        position_to_id[train_index], minlength=np.max(position_to_id) + 1
    )
    # Limit train frequencies to only those in the test set.
    train_frequencies = train_frequencies[test_class_idxs]

    test_frequencies = np.bincount(
        position_to_id[test_index], minlength=np.max(position_to_id) + 1
    )
    test_frequencies = test_frequencies[test_class_idxs]

    # Translate to vocab of word ID's.
    test_word_ids = class_to_test_idxs[position_to_id[test_index]]
    avg_auc, train_weighted_auc, test_weighted_auc = metrics.calculate_auc_roc(
        word_scores_np,
        test_word_ids,
        train_frequencies,
        test_frequencies,
        min_train_freq_auc,
        min_test_freq_auc,
    )
    results["test_word_avg_auc_roc"] = avg_auc
    results["test_word_train_weighted_auc_roc"] = train_weighted_auc
    results["test_word_test_weighted_auc_roc"] = test_weighted_auc

    for k_val in top_k_thresholds:
        results[f"test_word_top_{k_val}"] = metrics.top_k_accuracy(
            word_scores_np, test_word_ids, k_val
        )
    results["test_word_perplexity"] = metrics.perplexity(word_scores_np, test_word_ids)

    return results