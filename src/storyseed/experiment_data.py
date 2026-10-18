"""Pinned expanded training data and unchanged development/final populations."""
import json
import urllib.request
from pathlib import Path
import numpy as np
import torch
from storyseed.corpus import ROOT,REVISION,DATASET,SOURCE_FILE,DELIMITER,sha256,write_json,write_jsonl
from storyseed.story_splits import duplicate_key,split_name,load_split
from storyseed.story_tokenizer import StoryTokenizer,TOKENIZER_PATH
from storyseed.training_data import TrainingBatches

MANIFEST=ROOT/'data/manifests/phase07/corpus.json'
PROCESSED=ROOT/'data/processed/phase07'
PROTOCOL=ROOT/'planning/PHASE07_08_PROTOCOL.json'


def protocol():return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def encode_rows(rows,split,folder=PROCESSED):
    tokenizer=StoryTokenizer.load();ids=[];offsets=[0]
    for row in rows:
        sequence=tokenizer.encode(row['text'],add_bos=True,add_eos=True)
        ids.extend(sequence);offsets.append(len(ids))
    folder.mkdir(parents=True,exist_ok=True)
    token_path=folder/f'{split}_tokens.npy';offset_path=folder/f'{split}_offsets.npy'
    np.save(token_path,np.asarray(ids,dtype=np.uint16));np.save(offset_path,np.asarray(offsets,dtype=np.int64))
    return {'stories':len(rows),'prediction_positions':len(ids)-len(rows),
        'tokens':token_path.relative_to(ROOT).as_posix(),'offsets':offset_path.relative_to(ROOT).as_posix(),
        'tokens_sha256':sha256(token_path.read_bytes()),'offsets_sha256':sha256(offset_path.read_bytes())}


def prepare_corpus():
    spec=protocol()
    if MANIFEST.exists():
        saved=json.loads(MANIFEST.read_text(encoding='utf-8'))
        if saved['protocol_sha256']!=sha256(PROTOCOL.read_bytes()):raise ValueError('Existing expanded corpus belongs to a different protocol.')
        return saved
    count=spec['source_prefix_bytes']
    url=f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SOURCE_FILE}'
    request=urllib.request.Request(url,headers={'Range':f'bytes=0-{count-1}','User-Agent':'StorySeed-main/0.7'})
    with urllib.request.urlopen(request,timeout=45) as response:
        content_range=response.headers.get('Content-Range');status=response.status
        if status not in (200,206) or (status==206 and not (content_range or '').startswith('bytes 0-')):
            raise ValueError('Source response is not the requested pinned prefix.')
        prefix=response.read(count)
    if len(prefix)!=count:raise ValueError('Source prefix is shorter than the declared budget.')
    old=json.loads((ROOT/'data/manifests/phase02/source.json').read_text(encoding='utf-8'))
    if sha256(prefix[:old['prefix_bytes']])!=old['prefix_sha256']:raise ValueError('Expanded source disagrees with the original pinned prefix.')
    raw=ROOT/'data/raw/phase07';raw.mkdir(parents=True,exist_ok=True)
    (raw/'source-prefix.bin').write_bytes(prefix)
    groups={};duplicates=0
    parts=prefix.split(DELIMITER)
    for index,segment in enumerate(parts[:-1]):
        text=segment.decode('utf-8-sig').replace('\r\n','\n').strip()
        if not text:continue
        key=duplicate_key(text)
        if key in groups:duplicates+=1;continue
        groups[key]={'id':key,'source_index':index,'text':text,'content_sha256':sha256(text.encode('utf-8'))}
    training=[row for key,row in sorted(groups.items()) if split_name(key)=='train']
    # Read the already-known development text only. Final text is not loaded here.
    development=load_split('development')
    assignments=json.loads((ROOT/'data/manifests/phase02/splits.json').read_text(encoding='utf-8'))['assignments']
    forbidden={key for key,name in assignments.items() if name!='train'}
    if any(row['id'] in forbidden for row in training):raise ValueError('Expanded training contains an original reserved group.')
    old_train={key for key,name in assignments.items() if name=='train'}
    if not old_train.issubset({row['id'] for row in training}):raise ValueError('Original training population disappeared.')
    write_jsonl(PROCESSED/'train.jsonl',training)
    metadata={'format':'storyseed.expanded-corpus.v1','protocol_sha256':sha256(PROTOCOL.read_bytes()),
        'dataset':DATASET,'revision':REVISION,'url':url,'prefix_bytes':count,'prefix_sha256':sha256(prefix),
        'http_status':status,'content_range':content_range,'complete_unique_stories':len(groups),'duplicate_groups_removed':duplicates,
        'sampling':'convenience source prefix, not representative full TinyStories data',
        'training_rule':spec['training_selection'],'development_rule':spec['development_selection'],
        'tokenizer_sha256':sha256(TOKENIZER_PATH.read_bytes()),'original_split_manifest_sha256':sha256((ROOT/'data/manifests/phase02/splits.json').read_bytes()),
        'license_declared_by_pinned_card':old['license_declared_by_card'],'attribution':'Ronen Eldan and Yuanzhi Li, TinyStories (2023)',
        'files':{'train':encode_rows(training,'train'),'development':encode_rows(development,'development')},
        'normalized_original_reserved_overlap':0,'near_duplicate_limit':'normalized exact groups only; semantic duplicates not ruled out',
        'final_holdout_encoded':False,'training_jsonl_sha256':sha256((PROCESSED/'train.jsonl').read_bytes())}
    write_json(MANIFEST,metadata);return metadata


class ExperimentBatches(TrainingBatches):
    def __init__(self,split,context,score_tokens=64,manifest=MANIFEST):
        if not 1<=score_tokens<=context<=128:raise ValueError('Invalid experiment context.')
        metadata=json.loads(manifest.read_text(encoding='utf-8'))
        if metadata['tokenizer_sha256']!=sha256(TOKENIZER_PATH.read_bytes()):raise ValueError('Experiment tokenizer changed.')
        info=metadata['files'][split]
        for field in ['tokens','offsets']:
            path=ROOT/info[field]
            if sha256(path.read_bytes())!=info[field+'_sha256']:raise ValueError('Experiment arrays changed.')
            setattr(self,'ids' if field=='tokens' else 'offsets',np.load(path,mmap_mode='r'))
        self.locations=[(doc,start) for doc in range(len(self.offsets)-1) for start in range(0,int(self.offsets[doc+1]-self.offsets[doc])-1,score_tokens)]
        self.context,self.score_tokens=context,score_tokens
        self.target_counts=[min(score_tokens,int(self.offsets[doc+1]-self.offsets[doc])-1-start) for doc,start in self.locations]
        if sum(self.target_counts)!=info['prediction_positions']:raise ValueError('Target accounting mismatch.')


def experiment_identity(context):
    return {'protocol_sha256':sha256(PROTOCOL.read_bytes()),'corpus_manifest_sha256':sha256(MANIFEST.read_bytes()),
        'tokenizer_sha256':sha256(TOKENIZER_PATH.read_bytes()),'input_context':context,'score_tokens':64}
