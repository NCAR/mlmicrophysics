from sklearn.neighbors import KernelDensity
import numpy as np


def kde_sampled_pdf(data, sample_points, bandwidth=1.0, kernel="gaussian"):
    """
    Given a set of sampled values, create an empirical probability density function using kernel density
    estimation.

    Args:
        data: Array of samples from a distribution.
        sample_points: Array of points from which the pdf is sampled
        bandwidth: Bandwidth for the Kernel Density Estimate
        kernel: Type of kernel used. Default gaussian.

    Returns:
        Density of PDF at each point in sample_points
    """
    if len(data.shape) == 1:
        input_data = data.reshape(-1, 1)
    else:
        input_data = data
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(input_data)
    return np.exp(kde.score_samples(sample_points))


def hellinger_distance(y_true, y_pred, sample_points=np.linspace(-3, 3, 100), bandwidth=1.0, kernel="gaussian"):
    """
    Calculate the Hellinger Distance between two empirical distributions represented by their sampled values.

    Args:
        y_true: Array of true predictand values
        y_pred: Array of predicted predictand values
        sample_points: Array of points sampled from PDF for Hellinger distance
        bandwidth: Bandwidth of KDE
        kernel: Type of kernel used for KDE

    Returns:
        Hellinger distance (float)
    """
    pdf_true = kde_sampled_pdf(y_true, sample_points, bandwidth=bandwidth, kernel=kernel)
    pdf_pred = kde_sampled_pdf(y_pred, sample_points, bandwidth=bandwidth, kernel=kernel)
    hellinger = 1.0 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(pdf_true) - np.sqrt(pdf_pred)) ** 2))
    return hellinger



