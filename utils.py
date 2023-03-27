import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.close()


def visualize_eps_length(x, episode_durations, figure_file):
    plt.plot(x, episode_durations)
    plt.title('Episodes length')
    plt.savefig(figure_file)
    plt.close()


def plot_save_cdf(data_arr, save_name=None, xlabel=None):
    nbr_samples = data_arr.shape[0]
    unique, counts = np.unique(data_arr, return_counts=True)
    prob = counts/nbr_samples
    cdf = np.cumsum(prob)
    figure = plt.figure(figsize=(10, 10))
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.plot(unique, cdf)
    plt.savefig(f'data/{save_name}.png')

def plot_SEs(td3_sumSE, ref_sumSE, save_name=None,):
    assert len(td3_sumSE) == len(ref_sumSE), "TD3 and ref does not produce arrays with same shape"
    nbr_samples = len(td3_sumSE)
    figure = plt.figure(figsize=(10, 10))
    plt.xlabel("Number of test samples")
    plt.ylabel("SUM SE")
    plt.plot(range(nbr_samples), td3_sumSE, label='sun SE by TD3 agent')
    plt.plot(range(nbr_samples), ref_sumSE, label='sum SE by ref')
    plt.legend()
    plt.savefig(f'data/{save_name}.png')

def plot_SEs_CDF(td3_sumSE, ref_sumSE, save_name=None, xlabel=None):
    assert len(td3_sumSE) == len(ref_sumSE), "TD3 and ref does not produce arrays with same shape"
    nbr_samples = len(td3_sumSE)
    td3_unique, td3_counts = np.unique(td3_sumSE, return_counts=True)
    td3_prob = td3_counts/nbr_samples
    td3_cdf = np.cumsum(td3_prob)
    ref_unique, ref_counts = np.unique(ref_sumSE, return_counts=True)
    ref_prob = ref_counts/nbr_samples
    ref_cdf = np.cumsum(ref_prob)
    figure = plt.figure(figsize=(10, 10))
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.plot(td3_unique, td3_cdf, label='sum SE CDF by TD3 agent')
    plt.plot(ref_unique, ref_cdf, label='sum SE CDF by ref')
    plt.legend()
    plt.savefig(f'data/{save_name}.png')

def compare_results(td3_SE, geo_SE):
    td3_avg = np.average(td3_SE)
    print(f"Average SE from TD3 model: {td3_avg}")
    geo_avg = np.average(geo_SE)
    print(f"Average SE from geometric progarmming: {geo_avg}")
    # ratio = ((geo_avg - td3_avg) * 100) / geo_avg
    ratio = (td3_avg/geo_avg) * 100
    return ratio
