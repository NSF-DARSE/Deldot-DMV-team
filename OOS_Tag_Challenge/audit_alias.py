"""AUDIT: is the replaced name a consistent per-candidate alias across feeds?

If the generator gives each candidate the same fake name in every feed, the garbled row
is recoverable everywhere, not just on titles. Identify the garbled name on the title
feed (where vehicle_ref tells us the row belongs to a known candidate) and see whether
that exact name turns up in the other feeds more often than chance."""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from collections import defaultdict, Counter
from er import norm, sim
from pipeline import load
D=load(); cand=D['cand'].reset_index(drop=True); ttl=D['ttl'].copy()
ttl["f"]=ttl.owner_first_name.map(norm); ttl["l"]=ttl.owner_last_name.map(norm)

def clusters(names):
    m=len(names); lab=[-1]*m; k=0
    for i in range(m):
        if lab[i]>=0: continue
        lab[i]=k
        for j in range(i+1,m):
            if lab[j]<0 and 0.5*sim(names[i][0],names[j][0])+0.5*sim(names[i][1],names[j][1])>=0.62: lab[j]=k
        k+=1
    return lab

aliases=[]                       # (garbled_name, the 3-row majority name) per vehicle
for v,g in list(ttl.groupby("vehicle_ref"))[:6000]:
    if len(g)!=4: continue
    names=list(zip(g.f,g.l)); lab=clusters(names); cnt=Counter(lab)
    big=[k for k,c in cnt.items() if c==3]; sing=[k for k,c in cnt.items() if c==1]
    if len(big)==1 and len(sing)==1:
        gi=lab.index(sing[0]); bi=lab.index(big[0])
        aliases.append((names[gi], names[bi]))
print(f"vehicles with a clean 3+1 split: {len(aliases)}")

# how often does the garbled name appear in the other feeds?
pools={}
for k,(fc,lc) in [("adr",("first_name","last_name")),("ext",("first_name","last_name")),
                  ("lic",("first_name","last_name"))]:
    pools[k]=Counter(zip(D[k][fc].map(norm), D[k][lc].map(norm)))
real=Counter(zip(cand.first_name.map(norm), cand.last_name.map(norm)))

rs=np.random.RandomState(0)
fake=[a for a,_ in aliases]
# chance baseline: names built by recombining a random first and last name from the pool
allf=[a[0] for a,_ in aliases]; alll=[a[1] for a,_ in aliases]
sham=list(zip(rs.permutation(allf), rs.permutation(alll)))
print(f"\n{'feed':6s} {'garbled name found':>20s} {'recombined control':>20s}")
for k in ("adr","ext","lic"):
    hit=np.mean([pools[k][x]>0 for x in fake]); ctl=np.mean([pools[k][x]>0 for x in sham])
    print(f"{k:6s} {hit:19.1%} {ctl:19.1%}")
print(f"\ngarbled name is itself a real candidate's exact name: "
      f"{np.mean([real[x]>0 for x in fake]):.1%}")
first_real=np.mean([any(x[0]==c for c in cand.first_name.map(norm)) for x in fake[:400]])
print(f"its FIRST name alone belongs to some real candidate: {first_real:.1%}")

print("\n" + "="*74)
print("DECISIVE TEST: is an alias unique to one candidate, or shared by many?")
cnt=Counter(fake)
print("  vehicles sharing the same garbled name:", dict(sorted(Counter(cnt.values()).items())[:6]))
print(f"  distinct aliases: {len(cnt)} across {len(fake)} vehicles")

print("\nAre the alias rows the ones my matcher currently fails on?")
from resolve import Resolver, link_by_name
res=Resolver(cand)
adr=D["adr"]
ri,_,_=link_by_name(res,adr,"first_name","last_name")
linked=np.zeros(len(adr),bool); linked[ri]=True
key=list(zip(adr.first_name.map(norm),adr.last_name.map(norm)))
bykey=defaultdict(list)
for i,k in enumerate(key): bykey[k].append(i)
tot=hitrows=unl=0
for a in fake:
    rows=bykey.get(a,[])
    if rows:
        tot+=1; hitrows+=len(rows); unl+=sum(1 for r in rows if not linked[r])
print(f"  aliases present in the address feed: {tot}")
print(f"  address rows carrying an alias: {hitrows}; of those, currently UNLINKED: {unl} ({unl/max(hitrows,1):.1%})")
print(f"  baseline unlinked rate across the whole address feed: {(~linked).mean():.1%}")
