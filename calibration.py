import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from mapie.metrics.calibration import expected_calibration_error


def between_group_calibration(y_true, y_prob, sensitive_attribute, n_bins=2):
    """
    Calculates a fairness metric based on [2].

    Calibration as a boolean is defined as:
        P(Y=y | A=a, R=r) = P(Y=y | R=r) for all values r, a, and y

    We will define our metric as:
        |P(Y=y | A=a, R=r) - P(Y=y | A=b, R=r)|

    Assuming Y is binary, we'll just consider Y=1. Rather than enumerating all
        values of R, we'll bin and compute this difference within bins.  Then,
        we'll report the weighted average across bins.

    [2] https://dl.acm.org/doi/pdf/10.1145/3531146.3533245
    """

    n = y_true.shape[0]
    assert y_true.shape == y_prob.shape
    assert y_true.shape == sensitive_attribute.shape
    assert np.all((y_true == 0) | (y_true == 1))
    assert np.all((y_prob >= 0) & (y_prob <= 1))

    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)

    def metric(yt, yp):
        return np.mean(yt)
    metric_df = pd.DataFrame({"bin": bins, "A": sensitive_attribute})
    mf = MetricFrame(
        metrics=metric, y_true=y_true,
        y_pred=y_true, sensitive_features=metric_df)
    table = mf.by_group.unstack("A")
    gap = table.max(axis=1) - table.min(axis=1)
    counts = pd.Series(bins).value_counts()
    return float((gap * counts[gap.index] / len(y_true)).sum())


def test_ece():
    """
    Tests the expected calibration error function using the example from [1].

    Note that for this test, we're considering only binary classification,
    unlike the blog post. We've relabeled some examples and predictions so it
    works out to the same final answer.

    [1] https://iclr-blogposts.github.io/2025/blog/calibration/
    """

    y_prob = np.array([0.78, 0.36, 0.92, 0.58, 0.49, 0.85, 0.30, 0.63, 0.17])
    prob_arr = np.stack([1 - y_prob, y_prob], axis=1)

    y_true = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0])
    y_correct = ((y_prob > 0.5).astype(int) == y_true).astype(int)

    ece = expected_calibration_error(y_correct, prob_arr, num_bins=7)
    assert np.isclose(ece, 91/450), ece


def test_bgc():
    """
    Tests our between group calibration metric.
    """

    p = np.array([0.78, 0.36, 0.92, 0.58, 0.49, 0.85, 0.30, 0.63, 0.17])
    y = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0])
    a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])

    bgc = between_group_calibration(y, p, a, n_bins=2)
    assert np.isclose(bgc, 13/54), bgc


if __name__ == "__main__":
    test_ece()
    test_bgc()
