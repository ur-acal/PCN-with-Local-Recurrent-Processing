"""Parse original PT-only loss histories, deduplicating tqdm updates by batch."""
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
JOBS={'10_ideal':1724860,'10_0906':2091979,'10_1022':2205006,
      '100_ideal':1718361,'100_0906':2083946,'100_1022':2205011}
PATTERN=re.compile(r'Iter=(\d+)/390, Loss=([\d.e+-]+), CE=([\d.e+-]+), SRRL=([\d.e+-]+), LR=([\d.e+-]+), Stat=([\d.e+-]+), Pred=([\d.e+-]+)')


def parse(path):
    epochs=[];batches={};previous=0
    def flush():
        assert len(batches)==390,(path,len(epochs),len(batches))
        fields=['total','ce','srrl','lr','stat','pred']
        row={key:sum(v[j] for v in batches.values())/len(batches) for j,key in enumerate(fields)}
        row['total']=batches[390][0]
        row['epoch']=len(epochs)+1
        epochs.append(row)
    with path.open() as stream:
        for line in stream:
            if 'Train finished' in line:break
            for match in PATTERN.finditer(line):
                iteration=int(match[1])
                if iteration<previous:
                    flush();batches={}
                batches[iteration]=list(map(float,match.groups()[1:]))
                previous=iteration
    if batches:flush()
    assert len(epochs)==300,(path,len(epochs))
    return epochs


def main():
    report={}
    for key,job in JOBS.items():
        rows=parse(ROOT/'logs/slurm_jobs'/f'slurm_{job}.out')
        report[key]=dict(job=job,epochs=rows)
        print(key,[(end,round(sum(r['pred'] for r in rows[end-10:end])/10,4)) for end in [10,50,100,150,200,250,280,300]])
    (ROOT/'results/relu_srrl_gap/training_histories.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
