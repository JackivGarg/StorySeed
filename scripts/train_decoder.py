"""Run or resume the configured decoder; checkpoints are full training state."""
import argparse,json,time
from pathlib import Path
from storyseed.corpus import ROOT,write_json
from storyseed.training_config import TrainingConfig
from storyseed.training_runner import TrainingRunner
from storyseed.training_checkpoint import load_training,save_training
from storyseed.decoder_io import save_inference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--updates',type=int,help='Additional successful updates; default remaining configured budget.')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/phase06/latest.pt')
    parser.add_argument('--export',type=Path)
    parser.add_argument('--report',type=Path,default=ROOT/'runs/phase06/last-run.json')
    parser.add_argument('--max-seconds',type=float)
    parser.add_argument('--checkpoint-every',type=int,default=100)
    parser.add_argument('--evaluate-every',type=int,default=100,help='Successful-update interval; 0 disables periodic development scoring.')
    args=parser.parse_args()
    try:
        config=TrainingConfig.load(args.config) if args.config else None
        runner=load_training(args.resume,config) if args.resume else TrainingRunner(config or TrainingConfig.load(ROOT/'configs/phase06_teaching.json'))
        additional=args.updates if args.updates is not None else runner.config.total_updates-runner.updates
        if additional<0 or runner.updates+additional>runner.config.total_updates:raise ValueError('Requested updates exceed the saved schedule.')
        if args.max_seconds is not None and args.max_seconds<=0:raise ValueError('Positive time bound required.')
        if args.checkpoint_every<1 or args.evaluate_every<0:raise ValueError('Invalid checkpoint/evaluation intervals.')
        target=runner.updates+additional;started=time.monotonic();history=[]
        evaluations=[]
        max_attempts=runner.attempts+max(100,additional*4)
        while runner.updates<target:
            if args.max_seconds is not None and time.monotonic()-started>=args.max_seconds:break
            if runner.attempts>=max_attempts:raise RuntimeError('Too many skipped attempts; inspect numeric stability.')
            result=runner.step();history.append(result)
            print(json.dumps(result),flush=True)
            if not result['optimizer_step_skipped']:
                if runner.updates%args.checkpoint_every==0:save_training(runner,args.checkpoint)
                if args.evaluate_every and runner.updates%args.evaluate_every==0:
                    from storyseed.training_data import TrainingBatches
                    from storyseed.training_evaluation import evaluate_decoder
                    development=TrainingBatches('development',runner.config.model.context,runner.config.score_tokens)
                    metrics=evaluate_decoder(runner.model,development,runner.config.evaluation_batch_rows)
                    evaluations.append({'update':runner.updates,'metrics':metrics})
        artifact=save_training(runner,args.checkpoint)
        exported=save_inference(runner.model,args.export,'configured decoder training export') if args.export else None
        result={'completed_requested_updates':runner.updates==target,'requested_final_update':target,
            'actual_final_update':runner.updates,'history':history,'development_evaluations':evaluations,'checkpoint':artifact,'inference_export':exported,
            'elapsed_training_loop_seconds':time.monotonic()-started,'final_holdout_accessed':False}
        write_json(args.report,result)
        if runner.updates!=target:raise SystemExit(3)
    except (ValueError,RuntimeError,FileNotFoundError) as error:parser.error(str(error))

if __name__=='__main__':main()
