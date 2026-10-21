#!/usr/bin/env python3
"""Flag cache clips whose GT trajectory has a jump (a consecutive-frame step > 50x the median step, or
non-finite poses). Usage: scan_gt_jumps.py out.json"""
import numpy as np,glob,json,sys,zipfile
out={}
for ds in ["matrixcity","kitti360","tartanair-v1","tartanair-v2","tartanground","scannetpp","scannet-v2"]:
    root=f'/data/lingbot-cache1048-{ds}' if ds!='tartanground' else '/group/compact-3dmem/datasets/_staging/lingbot-cache1048-tartanground'
    bad=[];sc=[]
    for f in sorted(glob.glob(root+'/*.npz')):
        z=np.load(f); G=z['gt_c2w'].astype(np.float64); s=float(z['gt_scale'])
        st=np.linalg.norm(np.diff(G[:,:3,3],axis=0),axis=1)
        med=np.median(st)+1e-9
        jump=st.max()/med
        fin=np.isfinite(G).all()
        sc.append(s)
        if (not fin) or jump>50 or st.max()*s>5*max(1.0,np.percentile(st,99)*s*3):
            bad.append((f.split('/')[-1],round(s,2),round(float(jump),1),round(float(st.max()*s),2)))
    out[ds]=bad
    print(ds,'clips',len(sc),'flagged',len(bad),'gt_scale median',round(np.median(sc),2),'p99',round(np.percentile(sc,99),2), flush=True)
    for b in bad[:6]: print('   ',b, flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)
