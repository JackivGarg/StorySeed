"""Conditional paired-story uncertainty for fixed trained models."""
import numpy as np

def paired_bootstrap(a,b,replicates,seed):
    ac=np.asarray(a['story_target_counts']);bc=np.asarray(b['story_target_counts'])
    if not np.array_equal(ac,bc):raise ValueError('Paired comparison requires identical story target counts.')
    difference=np.asarray(a['story_loss_sums'])-np.asarray(b['story_loss_sums'])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(replicates):
        rows=rng.integers(len(ac),size=len(ac));values.append(float(difference[rows].sum()/ac[rows].sum()))
    return {'difference':'main128 minus context64; negative favors main128','observed_nll_difference':float(difference.sum()/ac.sum()),
        'story_bootstrap_95_percentile_interval':np.quantile(values,[.025,.975]).tolist(),'replicates':replicates,'seed':seed,
        'scope':'story-resampling uncertainty conditional on these two fixed trained models; not training-seed uncertainty or a broad superiority claim'}
