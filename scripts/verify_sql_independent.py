"""Recompute all 24 exercises from raw rows, without executing reference SQL.
Read-only: writes a new JSON report only. Usage: python -I script.py DATA OUTPUT
"""
import sys, json, gzip, sqlite3, hashlib
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

p, out = map(Path, sys.argv[1:])
if out.exists():
    raise SystemExit('Refusing to overwrite existing report')
c = sqlite3.connect(':memory:')
c.deserialize(gzip.decompress((p/'anticheat-demo.db.gz').read_bytes()))
c.row_factory = sqlite3.Row
T = {t: [dict(r) for r in c.execute('SELECT * FROM '+t)] for t in ['users','devices','riders','merchants','orders','coupons']}
c.close()
U = {x['user_id']: x for x in T['users']}
R = {x['rider_id']: x for x in T['riders']}
M = {x['merchant_id']: x for x in T['merchants']}
O = {x['order_id']: x for x in T['orders']}
def amt(x): return Decimal(str(x))
def rnd(x, n=2): return float(Decimal(x).quantize(Decimal(10)**-n, rounding=ROUND_HALF_UP))
def total(xs, k): return sum((amt(x[k]) for x in xs), Decimal(0))
def net(xs): return total(xs,'pay_amt')-total(xs,'refund_amt')
def dt(x): return datetime.fromisoformat(x)
def group(xs, f):
    d=defaultdict(list)
    for x in xs: d[f(x)].append(x)
    return d
week=[o for o in T['orders'] if '2026-10-26'<=o['order_ts']<'2026-11-02' and o['status']=='delivered']
a={}
a['Q01']=[[city,len(xs),rnd(net(xs))] for city,xs in group(week,lambda o:U[o['user_id']]['city']).items()]
a['Q01'].sort(key=lambda x:-x[2])
a['Q02']=[[rnd(net(week)/len(week))]]
a['Q03']=[[day,len(xs)] for day,xs in sorted(group([o for o in T['orders'] if '2026-10-01'<=o['order_ts']<'2026-11-01'],lambda o:o['order_ts'][:10]).items())]
a['Q04']=[[city,sum(o['is_new_user']==1 for o in xs),rnd(Decimal(sum(o['is_new_user']==1 for o in xs))/len(xs),4)] for city,xs in group(week,lambda o:U[o['user_id']]['city']).items()]
a['Q04'].sort(key=lambda x:-x[2])
cohort={u['user_id']:u for u in T['users'] if '2026-10-01'<=u['signup_ts']<'2026-10-26'}
first={o['user_id'] for o in T['orders'] if o['user_id'] in cohort and dt(cohort[o['user_id']]['signup_ts'])<=dt(o['order_ts'])<dt(cohort[o['user_id']]['signup_ts'])+timedelta(days=7)}
a['Q05']=[[len(cohort),len(first),rnd(Decimal(len(first))/len(cohort),4)]]
buyers=Counter(o['user_id'] for o in week); repeats=sum(n>=2 for n in buyers.values())
a['Q06']=[[len(buyers),repeats,rnd(Decimal(repeats)/len(buyers),4)]]
campaigns=group(T['coupons'],lambda x:x['campaign'])
a['Q07']=[[k,len(xs),sum(x['used_ts'] is not None for x in xs),rnd(Decimal(sum(x['used_ts'] is not None for x in xs))/len(xs),4)] for k,xs in campaigns.items()];a['Q07'].sort(key=lambda x:-x[3])
coupon_users={x['user_id'] for x in T['coupons'] if x['used_ts'] is not None and x['order_id'] in O and (dt(x['used_ts'])-dt(x['issued_ts'])).total_seconds()<60 and O[x['order_id']]['status']=='refunded'}
a['Q08']=[[len(coupon_users)]]
a['Q09']=[]
for k,xs in campaigns.items():
    orders=[O[x['order_id']] for x in xs if x['used_ts'] is not None and x['order_id'] in O]
    a['Q09'].append([k,len(orders),rnd(total(orders,'discount_amt')/total(orders,'gmv'),4)])
a['Q09'].sort(key=lambda x:-x[2])
devusers=defaultdict(set);userdevs=defaultdict(set)
for d in T['devices']:devusers[d['device_id']].add(d['user_id']);userdevs[d['user_id']].add(d['device_id'])
a['Q10']=sorted([[k,len(v)] for k,v in devusers.items() if len(v)>=3],key=lambda x:(-x[1],x[0]))
farm=devusers['D900001']|devusers['D900002'];a['Q11']=[[len(farm),sum(U[u]['signup_ts']>='2026-10-01' for u in farm)]]
ipdays=defaultdict(set)
for o in T['orders']:
    if o['order_ip']:ipdays[(o['order_ip'],o['order_ts'][:10])].add(o['user_id'])
a['Q12']=[[sum(len(v)>=5 for v in ipdays.values())]]
citydevices=group(T['users'],lambda u:u['city'])
a['Q13']=[[k,rnd(Decimal(sum(len(userdevs[u['user_id']]) for u in xs))/len(xs))] for k,xs in citydevices.items()];a['Q13'].sort(key=lambda x:-x[1])
cityriders=group(T['riders'],lambda r:r['city']);riderorders=Counter(o['rider_id'] for o in week)
a['Q14']=[[k,len(xs),sum(riderorders[r['rider_id']] for r in xs),rnd(Decimal(sum(riderorders[r['rider_id']] for r in xs))/len(xs))] for k,xs in cityriders.items()];a['Q14'].sort(key=lambda x:-x[3])
rg=group(T['orders'],lambda o:o['rider_id'])
a['Q15']=[[k,len(xs),rnd(Decimal(sum(o['status']=='cancelled' for o in xs))/len(xs),4)] for k,xs in rg.items() if len(xs)>=20];a['Q15'].sort(key=lambda x:(-x[2],-x[1]));a['Q15']=a['Q15'][:10]
pairs={(r['rider_id'],u) for r in T['riders'] if r['device_id'] for u in devusers[r['device_id']]}
a['Q16']=[[len(pairs)]]
a['Q17']=[[sum((o['rider_id'],o['user_id']) in pairs for o in T['orders'])]]
mg=group(T['orders'],lambda o:o['merchant_id'])
a['Q18']=[[k,len(xs),rnd(Decimal(sum(o['status']=='refunded' for o in xs))/len(xs),4)] for k,xs in mg.items() if len(xs)>=50];a['Q18'].sort(key=lambda x:(-x[2],-x[1]));a['Q18']=a['Q18'][:10]
a['Q19']=sorted([[k,rnd(total(xs,'refund_amt'))] for k,xs in mg.items()],key=lambda x:-x[1])[:5]
cat=group(T['orders'],lambda o:M[o['merchant_id']]['category']);gmv=total(T['orders'],'gmv')
a['Q20']=[[k,rnd(total(xs,'gmv')),rnd(total(xs,'gmv')/gmv,4)] for k,xs in cat.items()];a['Q20'].sort(key=lambda x:-x[1])
hours=Counter((o['user_id'],o['order_ts'][:13]) for o in T['orders']);hit={k:n for k,n in hours.items() if n>=8};vel={u for u,h in hit}
a['Q21']=[[len(vel),sum(hit.values())]]
refunds=Counter(o['user_id'] for o in T['orders'] if o['status']=='refunded');a['Q22']=[[sum(n>=3 for n in refunds.values())]]
multi=set().union(*(v for v in devusers.values() if len(v)>=3));a['Q23']=[[len(multi|coupon_users|vel)]]
a['Q24']=[]
for label,start,end in [('处置前','2026-10-18','2026-10-25'),('处置后','2026-10-25','2026-11-01')]:
    xs=[o for o in T['orders'] if o['user_id'] in farm and start<=o['order_ts']<end];a['Q24'].append([label,len(xs),len({o['user_id'] for o in xs})])
# Match complete rows, preserving duplicates. Numeric rounding is independently Decimal-based.
def equal(x,y):
    if isinstance(x,(int,float)) and isinstance(y,(int,float)):return abs(x-y)<1e-8
    return x==y
report=[]
for q in json.loads((p/'questions.json').read_text(encoding='utf-8')):
    got=a[q['id']];expected=q['golden_rows'];remaining=expected.copy();ok=len(got)==len(expected)
    for row in got:
        found=next((i for i,r in enumerate(remaining) if len(row)==len(r) and all(equal(x,y) for x,y in zip(row,r))),None)
        if found is None:ok=False;break
        remaining.pop(found)
    report.append({'id':q['id'],'matches_reference_values':ok,'independent_rows':got,'stored_rows':expected})
    print(q['id'], 'MATCH' if ok else 'DIFF', 'rows',len(got))
summary={'method':'Raw-row Python aggregation; no reference SQL execution; full result comparison ignoring row order','counts':{k:len(v) for k,v in T.items()},'data_sha256':hashlib.sha256((p/'anticheat-demo.db.gz').read_bytes()).hexdigest(),'cleaning':'None; all rows retained; metric-specific selections only','questions':report,'limitations':['ORDER BY and tie-breaking not independently certified; match checks values as multisets','Q13 assumes every user has at least one associated device; zero-device users must be explicitly checked','Q05 uses actual parsed timestamps rather than lexicographic SQLite text comparison']}
out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('TOTAL',len(report),'MATCH',sum(r['matches_reference_values'] for r in report),'DIFF',sum(not r['matches_reference_values'] for r in report))
print('zero-device users',sum(not userdevs[u] for u in U))
print('Report:',out)
