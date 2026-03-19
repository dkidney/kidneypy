from matplotlib.axes import Axes
import pandas as pd
from sklearn.model_selection._search import BaseSearchCV


def plot_cv_results(cls: BaseSearchCV, param: str, log_x: bool=False) -> Axes:
    cv_results = pd.DataFrame(cls.cv_results_)
    cols = cv_results.columns
    cols = cols[cols.str.startswith('param_')]
    cols = cols[~cols.isin([f'param_{param}'])]
    cv_results_best = cv_results.loc[[cls.best_index_], cols]
    df = cv_results.merge(cv_results_best, how='inner')
    col = f'param_{param}'
    ax = df.plot(x=col, y='mean_test_score', figsize=(9, 6), style='.-')
    ax.fill_between(df[col], 
                    df['mean_test_score'] - df['std_test_score'], 
                    df['mean_test_score'] + df['std_test_score'], alpha=0.15)
    ax.set_ylabel("mean_test_score")
    if log_x:
        ax.set_xscale("log")
        ax.set_xlabel(f'log({param})')
    else:
        ax.set_xlabel(param)
    ax.set_title(f'best score: {cls.best_score_}')
    ax.grid(True)
    return ax