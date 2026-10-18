import argparse,json
from pathlib import Path
import torch
from storyseed.corpus import ROOT
from storyseed.decoder_io import load_inference
from storyseed.context_generation import sample_context


def main():
    parser=argparse.ArgumentParser(description='Generate from a saved decoder inference bundle without fitting.')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/phase05/tiny-decoder.pt')
    parser.add_argument('--prompt',default='Once upon a time')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--temperature',type=float,default=0.8)
    parser.add_argument('--top-k',type=int,default=40)
    parser.add_argument('--max-new-tokens',type=int,default=48)
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    try:
        model,tokenizer=load_inference(args.checkpoint)
        result=sample_context(model,tokenizer,args.prompt,seed=args.seed,temperature=args.temperature,
            top_k=args.top_k,max_new_tokens=args.max_new_tokens)
    except (ValueError,RuntimeError,FileNotFoundError) as error:parser.error(str(error))
    print(json.dumps(result,indent=2,ensure_ascii=True) if args.json else json.dumps(result['text'],ensure_ascii=True)[1:-1])

if __name__=='__main__':main()
