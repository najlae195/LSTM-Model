from scipy import stats

def paired_ttest(a, b):
    t, p = stats.ttest_rel(a, b, nan_policy="omit")
    return float(t), float(p)
