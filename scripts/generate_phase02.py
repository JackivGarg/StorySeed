"""Load the Phase 2 tokenizer-compatible baseline without fitting again."""
import argparse
import json
from pathlib import Path
from storyseed.story_baseline import BASELINE_PATH, load_baseline, sample

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=BASELINE_PATH)
    parser.add_argument('--prompt',default='Once upon a time')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--temperature',type=float,default=1.0)
    parser.add_argument('--top-k',type=int)
    parser.add_argument('--max-new-tokens',type=int,default=48)
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    try:
        model,tokenizer=load_baseline(args.checkpoint)
        result=sample(model,tokenizer,args.prompt,seed=args.seed,temperature=args.temperature,
                      top_k=args.top_k,max_new_tokens=args.max_new_tokens)
    except (ValueError,FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result,indent=2,ensure_ascii=True) if args.json else json.dumps(result['text'], ensure_ascii=True)[1:-1])

if __name__=='__main__':main()
