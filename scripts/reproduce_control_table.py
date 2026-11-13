"""Recompute Table V means/ratios from recorded per-frame measurements (no inference)."""
from pathlib import Path
import csv,statistics
from decimal import Decimal,ROUND_HALF_UP
P=Path(__file__).resolve().parents[1]/'results/control'
with (P/'region_frames.csv').open() as f:q=list(csv.DictReader(f))
with (P/'matching_frames.csv').open() as f:m=list(csv.DictReader(f))
region='high_nir_rgb_usable'
fmt=lambda x,n:str(Decimal(str(x)).quantize(Decimal('1').scaleb(-n),rounding=ROUND_HALF_UP))
print('Method | Q scalar | Q color | SIFT count | precision % | yield % | ORB count | precision % | yield %')
for name,label in [('analytic_intensity','Analytic intensity'),('analytic_contrast','Analytic contrast'),('BF:v2b_lamina:fuser_ep08.pt','LaminaFuse')]:
 values=[]
 for interface in ['structure','color_Y']:
  rows=[r for r in q if r['method']==name and r['interface']==interface and r['region']==region]
  assert len(rows)==len({r['frame'] for r in rows})==96
  values.append(fmt(statistics.mean(float(r['q_source']) for r in rows),4))
 for extractor in ['SIFT','ORB']:
  rows=[r for r in m if r['method']==name and r['extractor']==extractor and r['region']==region]
  assert len(rows)==len({r['frame'] for r in rows})==96
  c=sum(int(r['correct_3px']) for r in rows)
  values += [fmt(c/96,2),fmt(100*c/sum(int(r['accepted_matches']) for r in rows),2),fmt(100*c/sum(int(r['supported_left']) for r in rows),2)]
 print(label+' | '+' | '.join(values))
