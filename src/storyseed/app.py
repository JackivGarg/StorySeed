"""One command for StorySeed's local application."""
import argparse
import json
from pathlib import Path

from storyseed.app_service import GenerationRequest, GenerationService


def main():
    parser = argparse.ArgumentParser(description='Continue a story using the declared scratch-trained StorySeed release.')
    parser.add_argument('--prompt', default='Once upon a time')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--max-new-tokens', type=int, default=192)
    parser.add_argument('--json', action='store_true', help='Print generation settings and artifact identity with the text.')
    parser.add_argument('--output', type=Path, help='Save the complete result as a UTF-8 JSON file.')
    parser.add_argument('--gui', action='store_true', help='Open the local desktop interface (requires Tkinter).')
    parser.add_argument('--original', action='store_true', help='Use the historical 512-vocabulary main model for comparison.')
    parser.add_argument('--bundle', type=Path, help='Load a portable StorySeed release folder instead of repository artifacts.')
    args = parser.parse_args()
    if args.original and args.bundle:
        parser.error('Choose either --original or --bundle.')
    def service_factory():
        if args.original:
            from storyseed.corpus import ROOT
            return GenerationService(ROOT / 'checkpoints/phase07/main128-inference.pt')
        if args.bundle:
            from storyseed.release_bundle import load_bundle
            return load_bundle(args.bundle)
        return GenerationService()
    if args.gui:
        if args.output or args.json:
            parser.error('--output/--json apply to command-line generation; use Save result inside the window.')
        try:
            from storyseed.desktop import launch
            launch(service_factory)
        except (ImportError, RuntimeError) as error:
            parser.error(f'Desktop interface unavailable: {error}. Use the command line or install Python with Tcl/Tk.')
        return
    request = GenerationRequest(args.prompt, args.seed, args.temperature, args.top_k, args.max_new_tokens)
    try:
        request.validate()
        result = service_factory().generate(request)
        serialized = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding='utf-8')
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(serialized if args.json else result['text'], end='' if args.json else '\n')


if __name__ == '__main__':
    main()
