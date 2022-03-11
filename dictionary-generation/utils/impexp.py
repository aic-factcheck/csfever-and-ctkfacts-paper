from collections import OrderedDict
import json
import pathlib
from pathlib import Path

# from fever-baselines
class Reader:
    def __init__(self,encoding="utf-8"):
        self.enc = encoding

    def read(self,file):
        with open(file,"r",encoding = self.enc) as f:
            return self.process(f)

    def process(self,f):
        pass

class JSONLineReader(Reader):                                                                                                                                                  
    def process(self,fp):                                                                                                                                                      
        data = []                                                                                                                                                              
        for line in fp.readlines():                                                                                                                                            
            data.append(json.loads(line.strip()))                                                                                                                              
        return data   

def read_jsonl(jsonl):
    with open(jsonl, 'r') as json_file:
        json_list = json_file.read()
        data = [json.loads(jline, object_pairs_hook=OrderedDict) for jline in json_list.splitlines()]
    return data

def write_jsonl(jsonl, data):
    # data is an iterable (list) of JSON-compatible structures (OrderedDict)
    with open(jsonl, 'w', encoding='utf8') as json_file:
        for r in data:
            json.dump(r, json_file, ensure_ascii=False)
            json_file.write("\n")
            
def read_json(fname, object_pairs_hook=OrderedDict):
    with open(fname, 'r') as json_file:
        data = json.load(json_file, object_pairs_hook=object_pairs_hook)
    return data

def write_json(fname, data, indent=3):
    with open(str(fname), 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=indent)
        
def to_tuples(obj):
    # recursively converts lists, dicts and sets in json-like structures to tuples
    # dicts and sets are sorted by keys
    # note: type information is lost
    if isinstance(obj, (list, tuple)):
        return tuple(to_tuples(e) for e in obj)
    elif isinstance(obj, set):
        return tuple(to_tuples(e) for e in sorted(obj))
    elif isinstance(obj, dict):
        return tuple(to_tuples(e) for e in sorted(obj.items()))
    else:
        return obj
    
def merge_jsonl_records(a, b, equal_keys=None):
    # the records of `b` are appended to `a` in the order of `b` (if not already present in `a`)
    # if present, they replace `a` items
    # if `equal_keys` is None the records are compared in original form
    # `equal_keys` can be a set/list of root-level keys to be compared only
    if equal_keys is not None:
        a2 = [{k: e[k] for k in equal_keys} for e in a]
        b2 = [{k: e[k] for k in equal_keys} for e in b]
    else:
        a2, b2 = a, b
        
    adict = OrderedDict([(e, i) for i, e in enumerate(to_tuples(a2))]) # hash a so we can search it fast, also store original index
    cnt = 0
    for i, brec in enumerate(to_tuples(b2)):
        if brec not in adict:
            cnt += 1
            a.append(b[i])
        else:
            a[adict[brec]] = b[i] # replace with newer
    print(f"new records: {cnt}/{len(b2)}")

def read_jsonl_dir(jsonl_dir):
    # loads all jsonl files in sortred order in `jsonl_dir` while merging them
    merged=None
    for f in sorted(Path(jsonl_dir).glob("*.jsonl")):
        jsonl = read_jsonl(f)
        if merged is None:
            merged = jsonl
        else:
            merge_jsonl_records(merged, jsonl, equal_keys=["id"])
    return merged