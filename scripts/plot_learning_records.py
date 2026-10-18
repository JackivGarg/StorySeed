"""Plot existing observations only; this script does not train or evaluate a model."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True,help='Separate study-notes folder')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    tiny=json.loads((root/'reports/phase05/06_tiny_learning.json').read_text(encoding='utf-8'))
    observations=json.loads((root/'reports/phase06/08_learning_observations.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    assets=[]

    def save(fig,phase,name):
        folder=args.output_root/phase/'assets'
        folder.mkdir(parents=True,exist_ok=True)
        for suffix in ['png','svg']:
            path=folder/f'{name}.{suffix}'
            fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
            assets.append({'path':path.relative_to(args.output_root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)

    x=[point['step']-1 for point in tiny['minibatch_history']]+[tiny['updates']]
    y=[point['pre_update_loss'] for point in tiny['minibatch_history']]+[tiny['final_tiny_batch_nll']]
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.plot(x,y,'o-',color='#166b83',label='Recorded observations; same 124 targets')
    ax.set_yscale('log');ax.set_xlabel('Optimizer updates already completed');ax.set_ylabel('Mean NLL (nats per target; log scale)')
    ax.set_title('Phase 5: deliberate tiny-batch memorization',loc='left',fontweight='bold')
    ax.annotate(f'{y[0]:.4f}',(x[0],y[0]),xytext=(9,-10),textcoords='offset points')
    ax.annotate(f'{y[-1]:.6f} after 200 updates',(x[-1],y[-1]),xytext=(-190,20),textcoords='offset points')
    ax.grid(alpha=.18,which='both');ax.legend(loc='upper right',frameon=False)
    fig.text(.12,-.02,'Sparse recorded points only. This is training-batch fitting, not development quality.',fontsize=10,color='#555555')
    save(fig,'phase05','tiny-batch-loss')

    history=observations['training_observations']
    fig,(ax,lr)=plt.subplots(2,1,figsize=(9,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    updates=[point['update'] for point in history]
    ax.plot(updates,[point['nll'] for point in history],'o-',color='#166b83')
    ax.set_ylabel('Pre-update batch NLL');ax.grid(alpha=.18)
    ax.set_title('Phase 6: ten sampled effective batches',loc='left',fontweight='bold')
    ax.text(.02,.08,'Different rows and dropout masks at each point.\nThese are not paired evaluations.',transform=ax.transAxes,fontsize=10)
    lr.plot(updates,[point['learning_rate'] for point in history],'o-',color='#bc6b30')
    lr.set_xlabel('Associated successful update (loss measured before the update)');lr.set_ylabel('Learning rate');lr.grid(alpha=.18);lr.set_xticks(updates)
    fig.tight_layout();save(fig,'phase06','pilot-training-loss')

    dev=observations['development_observations']
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.scatter([point['completed_updates'] for point in dev],[point['nll'] for point in dev],s=85,color='#166b83')
    for point in dev:
        ax.annotate(f"NLL {point['nll']:.6f}\nPPL {point['perplexity']:.3f}",(point['completed_updates'],point['nll']),xytext=(0,12),textcoords='offset points',ha='center')
    ax.set_xlim(-3,13);ax.set_ylim(5.65,6.55);ax.set_xticks([0,10]);ax.set_xlabel('Successful updates completed');ax.set_ylabel('Full development NLL (nats per target)')
    ax.set_title('Same 39,711 development targets; two observations',loc='left',fontweight='bold');ax.grid(alpha=.18)
    fig.text(.13,-.01,'No intermediate development observations were recorded.',fontsize=10,color='#555555')
    save(fig,'phase06','development-loss')
    print(json.dumps({'source':'existing report observations; no fitting or evaluation performed','assets':assets},indent=2))


if __name__=='__main__':main()
