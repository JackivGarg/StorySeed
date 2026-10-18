"""One command for StorySeed's local application."""
import argparse
import json
from pathlib import Path

from storyseed.app_service import GenerationRequest, GenerationService


def main():
    parser = argparse.ArgumentParser(description='Continue a story using the fixed scratch-trained StorySeed main model.')
    parser.add_argument('--prompt', default='Once upon a time')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--max-new-tokens', type=int, default=48)
    parser.add_argument('--json', action='store_true', help='Print generation settings and artifact identity with the text.')
    parser.add_argument('--output', type=Path, help='Save the complete result as a UTF-8 JSON file.')
    args = parser.parse_args()
    request = GenerationRequest(args.prompt, args.seed, args.temperature, args.top_k, args.max_new_tokens)
    try:
        request.validate()
        result = GenerationService().generate(request)
        serialized = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding='utf-8')
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(serialized if args.json else result['text'], end='' if args.json else '\n')


if __name__ == '__main__':
    main()
