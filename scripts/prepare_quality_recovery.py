import json
from storyseed.quality_data import prepare

if __name__ == '__main__':
    result = prepare()
    print(json.dumps(result, indent=2), flush=True)
