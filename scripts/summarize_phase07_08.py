"""Render actual completed run records into study figures and a clearly scoped draft."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from storyseed.corpus import ROOT,sha256
from storyseed.experiment_data import write_json

NOTES=Path('C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes')


def main():
    main_run=json.loads((ROOT/'reports/phase07/main128.json').read_text(encoding='utf-8'))
    short=json.loads((ROOT/'reports/phase07/context64.json').read_text(encoding='utf-8'))
    corpus=json.loads((ROOT/'data/manifests/phase07/corpus.json').read_text(encoding='utf-8'))
    metrics=json.loads((ROOT/'reports/phase08/metrics.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    assets=[]
    for run in [main_run,short]:
        name=run['condition']['id'];curve=json.loads((ROOT/f'reports/phase07/{name}-curve.json').read_text(encoding='utf-8'))
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        x=[point['update']-1 for point in curve['updates']]
        axes[0].plot(x,[point['nll'] for point in curve['updates']],alpha=.4,lw=.7,color='#176b83',label='sampled training NLL, before update')
        dev=curve['development']
        axes[0].plot([point['completed_updates'] for point in dev],[point['development']['nats_per_target_token'] for point in dev],'o-',color='#bc6b30',label='full development, dropout off')
        axes[0].set_ylabel('Nats per true target');axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
        axes[0].set_title(f'{name}: actual scratch-training observations',loc='left',fontweight='bold')
        axes[1].plot([point['update'] for point in curve['updates']],[point['learning_rate'] for point in curve['updates']],color='#176b83')
        axes[1].set_xlabel('Completed updates (development) / completed updates before forward (training)');axes[1].set_ylabel('Learning rate');axes[1].grid(alpha=.2)
        fig.tight_layout();folder=NOTES/'phase07/assets';folder.mkdir(parents=True,exist_ok=True)
        for suffix in ['png','svg']:
            path=folder/f'{name}-learning.{suffix}';fig.savefig(path,dpi=180,bbox_inches='tight');assets.append({'path':path.relative_to(NOTES).as_posix(),'sha256':sha256(path.read_bytes())})
        plt.close(fig)
    names=list(metrics['models'])
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(names,[metrics['models'][name]['nats_per_target_token'] for name in names],color=['#176b83','#4f98ac','#bc6b30','#b7ada4'])
    ax.set_xlabel('Mean NLL per token on the frozen final population (lower is better)');ax.set_title('Actual final-set scores; baseline training budgets differ',loc='left',fontweight='bold')
    for i,name in enumerate(names):ax.text(metrics['models'][name]['nats_per_target_token']+.02,i,f"{metrics['models'][name]['nats_per_target_token']:.4f}",va='center')
    fig.tight_layout();folder=NOTES/'phase08/assets';folder.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg']:
        path=folder/f'final-loss.{suffix}';fig.savefig(path,dpi=180,bbox_inches='tight');assets.append({'path':path.relative_to(NOTES).as_posix(),'sha256':sha256(path.read_bytes())})
    plt.close(fig)
    write_json(ROOT/'reports/phase08/chart_assets.json',assets)
    table='| Model | Final NLL | Perplexity | Target positions |\n|---|---:|---:|---:|\n'
    for name in names:
        value=metrics['models'][name];table+=f"| {name} | {value['nats_per_target_token']:.6f} | {value['perplexity']:.3f} | {value['prediction_positions']} |\n"
    dev=main_run['development_curve'];first=dev[0]['development']['nats_per_target_token'];last=dev[-1]['development']['nats_per_target_token']
    phase7=f'''# Observed main-training outcome

Generated from completed records; no values are projected. The main128 model began from random weights, with {main_run['parameter_ledger']['total']:,} parameters, and completed {main_run['successful_updates']:,} successful updates. Training uses {corpus['files']['train']['stories']:,} stories from the pinned 8 MiB source prefix and the unchanged 512-entry tokenizer.

Full development NLL moved from {first:.6f} to {last:.6f} on the same {dev[-1]['development']['prediction_positions']:,} targets. The training process presented {main_run['scored_target_presentations']:,} true targets, equivalent to {main_run['target_presentations_per_population']:.3f} times the available training target population. Sampling is with replacement, so that ratio is not proof that every position was visited.

![Actual main learning records](assets/main128-learning.png)

Read the curve by its measurement contract: the fine training trace uses changing sampled rows with dropout before each update. The development markers use a fixed population, evaluation mode and completed-update weights. An upward training point need not mean the model forgot; a persistent development rise needs investigation. The individual recorded markers, not a smooth invented trajectory, support interpretation.

Learning changes the parameters through accumulated token gradients and AdamW. The architecture, tokenizer and underlying labels stay fixed during each run. No component ablation isolates how much of this improvement came from a particular head, sublayer or schedule. Inspect actual samples in the curve record to connect predictive improvement to visible generation failures.

The exported artifact is `{main_run['inference']['path']}`, SHA-256 `{main_run['inference']['sha256']}`. The public generator reproduces its seeded output in a fresh process; it does not train. Full resume checkpoints contain optimizer/random/data state, while the inference export does not.

The companion context64 run is a separate random-initialized training condition, not a shortened view of the already-trained main model. Both runs are used by the frozen Phase 8 protocol. Learner understanding remains pending.
'''
    (NOTES/'phase07/OBSERVED_RESULTS.md').write_text(phase7,encoding='utf-8')
    interval=metrics['paired_story_comparison']['story_bootstrap_95_percentile_interval']
    phase8=f'''# Observed fixed-protocol evaluation

The final population was encoded only after both predeclared 2,000-update model exports were fixed. No additional training or selection used these results.

{table}
![Actual frozen final loss](assets/final-loss.png)

The paired context difference is main128 minus context64: {metrics['paired_story_comparison']['observed_nll_difference']:.6f} nats/target. Its descriptive story-bootstrap 95% percentile interval is [{interval[0]:.6f}, {interval[1]:.6f}]. Negative values favor main128 under the stated supplied-context policy. The resampling holds trained weights fixed and varies stories; it does not estimate uncertainty across training seeds. There is only one paired training seed.

Equal parameter count, initial model state, sampled training rows, successful updates, true-target presentations and final target counts were checked in the recorded execution. Compute is not matched: longer inputs require more processing. Window-relative positions and dropout draws change with supplied input length, so this is a comparison of context policies in our implemented system.

The expanded count bigram sees exactly the expanded training population and uses fixed additive smoothing alpha 0.1. The historical Phase 2 learned bigram used less training data and a different optimization budget. Their scores are useful references, not an architecture-only capacity-matched experiment.

The paired continuation packet contains 24 generated samples from six frozen prompts and two sampling seeds per condition. Word-trigram repetition and lexical diversity measure observable surface properties. They do not measure narrative coherence, and their interpretation depends on output length and decoding settings. Human rating fields remain blank and pending. A future assistant review must identify itself rather than being described as an independent human study.

See the actual `reports/phase08/metrics.json`, `samples.json`, `PAIRED_SAMPLE_REVIEW.md` and `PAPER_DRAFT.md` in the code repository. The final set is now used under this frozen protocol; future tuning needs a new untouched evaluation population. Learner understanding remains pending.
'''
    (NOTES/'phase08/OBSERVED_RESULTS.md').write_text(phase8,encoding='utf-8')
    paper=f'''# StorySeed: A Resource-Constrained Study of Context Policies in a Scratch-Trained Small Language Model

Status: university research draft based on actual local execution. Independent human evaluation is pending. No venue submission or acceptance is claimed.

## Abstract

We implement a compact causal decoder explicitly in PyTorch and train it from random initialization for simple English story continuation. We compare 64 versus 128 supplied input slots while holding maximum position-table size, parameter count, initial weights, training row draws and successful target/update budgets fixed. The experiment uses one paired training seed and a bounded TinyStories source prefix. We report full-vocabulary next-token NLL, perplexity, lexical repetition, measured resources and a paired review packet. The study demonstrates a reproducible educational experiment, rather than a new language-model architecture.

## 1. Motivation and related work

Transformers introduced attention-based sequence processing [1]. TinyStories studies language learning in much smaller models using constrained synthetic stories [2]. StorySeed follows established techniques and asks a local resource/conditioning question: how do two supplied-context policies behave under an equal target budget? It does not claim to reproduce the original paper's training scale or results.

## 2. Data and implementation

We capture the first 8 MiB of TinyStories-train.txt at revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64. Training uses {corpus['files']['train']['stories']:,} normalized-unique groups assigned train by the unchanged seed-17 content-hash rule. The original 105 development and 120 final stories remain separate. The unchanged 512-entry byte-level BPE tokenizer was fit using only the earlier training subset. This small vocabulary increases sequence fragmentation and is a limitation, not a claim of an optimal tokenizer. Exact/normalized overlap checks do not rule out semantic near duplicates. Source-prefix sampling is not representative of the full dataset.

The model has {main_run['parameter_ledger']['total']:,} parameters: width 256, eight heads, eight pre-LayerNorm decoder blocks, GELU feed-forward width 1,024, learned 128-position embeddings and untied vocabulary weights. Parameters begin randomly. PyTorch supplies differentiation/device infrastructure; no pretrained weights are used.

## 3. Predeclared method

Each condition completes 2,000 AdamW updates with four-row microbatches accumulated four times, scored width 64, warm-up 100, maximum LR 0.0003 and minimum LR 0.00003. Both use the same seed 17, data seed 2026, dropout 0.1 and successful target budget. CUDA BF16 autocast uses FP32 master weights and loss. The last checkpoint is selected in advance; development is measured every 250 updates. A final set is encoded/scored only after both exports are frozen.

With 64 slots, a row supplies up to its 64 scored inputs. With 128 slots, up to 64 additional earlier story inputs are prepended. This policy does not provide a separately maximal prefix at every target. Initial weights, rows and target counts match; compute does not. Window-relative position usage and dropout realizations vary with input length. This is an intervention on the implemented context policy, not a universal architectural claim.

## 4. Results

{table}
Main full-development NLL changes from {first:.6f} to {last:.6f}. This is a fixed-population predictive measurement, distinct from noisy pre-update training batches. The paired final NLL difference is {metrics['paired_story_comparison']['observed_nll_difference']:.6f}; descriptive paired story-bootstrap interval [{interval[0]:.6f}, {interval[1]:.6f}]. This interval conditions on one trained pair and does not capture training-seed variability.

Main measured update compute is {main_run['measured_update_compute_seconds']:.3f} seconds; short-context compute is {short['measured_update_compute_seconds']:.3f} seconds. Full elapsed and peak-memory records are reported separately in each condition JSON; update timing excludes periodic evaluation/checkpointing. Throughput is scored target IDs per measured update second, not words per second.

Twelve sampled continuations per condition use six predeclared prompts and seeds 17/29, temperature 0.8, top-k 40 and maximum 128 new IDs. Individual samples, EOS/length and lexical metrics are preserved. Grammar/continuity/consistency judgments require review; no human scores are supplied here.

## 5. Limitations

The dataset is a small convenience prefix, the tokenizer was learned on a smaller training subset, and there is one paired training seed. Hyperparameter choices are not claimed optimal. The historical baseline differs in data exposure/budget; the count baseline has a different inductive bias/fit procedure. Word repetition is not narrative coherence. The fixed held-out evaluation must not be reused for further tuning. No independent human ratings, agreement analysis, broad benchmark superiority, new algorithm or publication acceptance is claimed.

## 6. Conclusion

The project produces traceable scratch-trained weights and a controlled local comparison with honest limits. The evidence supports the measured conditional findings above and provides a basis for additional seeds, independent story review and larger-data work. It does not establish general superiority from one run.

## References

[1] Vaswani et al. (2017), Attention Is All You Need. https://arxiv.org/abs/1706.03762

[2] Eldan and Li (2023), TinyStories: How Small Can Language Models Be and Still Speak Coherent English? https://arxiv.org/abs/2305.07759

[3] Pinned TinyStories dataset card (source/declared terms): https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md

[4] Perplexity and finite-context evaluation: https://huggingface.co/docs/transformers/perplexity

## Reproduction and artifacts

The protocol and its byte hashes are in planning/PHASE07_08_PROTOCOL.json and PHASE07_08_FREEZE.json. Source/configuration, provenance, split/tokenizer identities, condition reports, full curves, final metrics, generated samples and review forms are versioned; raw text and tensor checkpoints remain local and ignored. The main export SHA-256 is `{main_run['inference']['sha256']}`. The finite foreground command is `.venv/Scripts/python scripts/run_phase07_08.py`; re-executing it restores valid unfinished states or reads completed compatible records, rather than silently changing the protocol.
'''
    operational='''
## Operational timing incident and recovery

The context64 process stopped at a valid checkpoint after its runtime guard fired. One recorded update contained approximately 17,768 seconds of synchronized wall delay, while the median preceding update was approximately 0.182 seconds. The cause of that isolated system/device stall was not established. The execution allowance was explicitly amended before final-set access; data, optimizer settings, RNG checkpoint, target/update budgets and last-artifact selection stayed fixed. The unfinished condition resumed from update 669.

Raw timing retains the stall; gross condition durations must not be interpreted as a clean context-speed comparison. A separate matched resource-only observation uses scratch copies, fixed rows, three trials per context, three warm-up and ten timed updates per trial. All samples are retained, and an update above the predeclared five-second threshold marks that observation contaminated. See matched_resources.json for actual validity and measurements. This resource observation does not change the frozen trained artifacts or their quality scores.
'''
    paper+=operational;phase7+=operational;phase8+=operational
    with (NOTES/'phase07/OBSERVED_RESULTS.md').open('a',encoding='utf-8') as stream:stream.write(operational)
    with (NOTES/'phase08/OBSERVED_RESULTS.md').open('a',encoding='utf-8') as stream:stream.write(operational)
    (ROOT/'reports/phase08/PAPER_DRAFT.md').write_text(paper,encoding='utf-8')
    (ROOT/'reports/phase07/SUMMARY.md').write_text(phase7.replace('(assets/','(C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes/phase07/assets/'),encoding='utf-8')
    (ROOT/'reports/phase08/SUMMARY.md').write_text(phase8.replace('(assets/','(C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes/phase08/assets/'),encoding='utf-8')
    print(json.dumps({'plots':len(assets),'paper_draft':'reports/phase08/PAPER_DRAFT.md','human_review':'pending'}))


if __name__=='__main__':main()
