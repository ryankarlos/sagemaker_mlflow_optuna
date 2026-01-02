"""Evaluation metrics for FM pipeline."""

import numpy as np


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def dcg_score(score_vector, k):
    """
    y_true=[10,10,100,0,0,0,0,0,0,0]
    y_score=[10,0,0,0,0,0,0,0,0,0]
    actual=dcg_score(y_score,2)
    best=dcg_score(y_true,2)
    """
    trimmed_score_vector = score_vector[0 : min(len(score_vector), k)]
    gain = trimmed_score_vector
    discounts = np.log2(np.arange(len(trimmed_score_vector)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(y_score, y_true, k):
    actual = dcg_score(score_vector=y_score, k=k).astype(np.float64)
    best = dcg_score(score_vector=y_true, k=k).astype(np.float64)
    if best:
        return actual / best
    else:
        return np.nan


def get_ndcg_scores(
    actual_preference_item_tuple: list, predict_user_rec_items: list, k=10
):
    """
    gets NDCG score based on either K or if the number of actuals lower than on that.

    we get the
    :param actual_preference_item_tuple: list of tuples (preference,item) for a user in descending order (number of users X number of actual items for the user)
    :param predict_user_rec_items: list of items that a user was recommended with (n x predicted items)
    :param k: number of items to consider
    :return: ndgc scores by user
    """

    # print("actual_preference_item_tuple: ", len(actual_preference_item_tuple))
    # print("predict_user_rec_items: ", len(predict_user_rec_items))

    # evaluation
    scores = []
    for idx, rec_items in enumerate(predict_user_rec_items):
        user_rec_relevance = np.zeros(len(rec_items))  # user x no of rec items
        preference_items = actual_preference_item_tuple[idx]
        item_relevances, items = zip(*preference_items, strict=False)
        item_relevances = np.pad(item_relevances, (0, max(k - len(item_relevances), 0)))
        rec_items = rec_items[0:k]

        for pidx, pi in enumerate(items):
            position_list = np.where(rec_items == pi)[
                0
            ]  # return the position of the item in the recommended list
            if len(position_list) > 0:
                rec_position = position_list[0]
                item_relevance = item_relevances[pidx]  # gain
                user_rec_relevance[rec_position] = (
                    item_relevance  # write gain to user x item rec matrix
                )

        y_score = user_rec_relevance
        y_true = np.zeros_like(y_score)
        y_true[: len(rec_items)] = item_relevances[: len(rec_items)]
        score = ndcg_score(y_true=y_true, y_score=y_score, k=k)
        scores.append(score)

    return scores
