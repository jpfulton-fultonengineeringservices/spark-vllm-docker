#!/usr/bin/env python3
"""b12x-native lut_e4m3 round-trip selfcheck.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os; os.environ.setdefault("LD_LIBRARY_PATH", "/usr/local/lib/python3.12/dist-packages/torch/lib")
import sys
import time
from pathlib import Path
from typing import Any

_TORCH=None
def _rt():
    global _TORCH
    if _TORCH is None:
        import torch as t; _TORCH=t
    return _TORCH

def _r(c,ok,**m): return {"check":c,"ok":bool(ok),**m}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-out", type=Path, required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--num-experts", type=int, default=1)
    ap.add_argument("--bits", type=int, default=3, choices=(2,3,4))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    try: torch = _rt()
    except Exception as e:
        print(json.dumps({"checks":[_r("imports",False,reason=str(e))]},indent=2))
        return 1
    sys.path.insert(0,"/tmp")

    rpt={"args":{k:str(v) for k,v in vars(args).items()},"checks":[]}
    K,N,E=args.K,args.N,args.num_experts
    assert K%16==0 and N%32==0

    # synthetic weight
    W=torch.zeros(N,K,dtype=torch.float32)
    for n in range(N):
        nw=n%16
        for k in range(K):
            W[n,k]=(k//16)*0.1+nw*1.0+(k%16)*0.01
    rpt["checks"].append(_r("W",True,shape=list(W.shape),rms=float(W.pow(2).mean().sqrt())))

    from b12x_native_quantizer import encode_expert_lut_e4m3
    enc=encode_expert_lut_e4m3(W,bits=args.bits,is_fc2=False,device="cpu")
    rpt["checks"].append(_r("encode",True,**{
        "K_tiles":enc["K_tiles"],"num_slots":enc["num_slots"],"bits":enc["bits"],
        "p0l_sha":hashlib.sha256(enc["planes"][(0,"low")].numpy().tobytes()).hexdigest()[:16],
    }))

    from b12x.moe._shared.kernels.w4a16.exl3_synth import (
        Exl3SynthConfig,synth_layer_payloads,Exl3LayerPayloads,
        assemble_code_rows,layer_metadata,_manifest_dict,
    )
    from b12x.moe._shared.exl3_schema import layer_filename,EXL3_MANIFEST_FILENAME
    from safetensors.torch import save_file
    LUT_E4M3="lut_e4m3"

    cfg=Exl3SynthConfig(
        codebook=LUT_E4M3,num_experts=E,hidden_size=K,
        intermediate_size=N,moe_layer_indices=(0,),bits=args.bits,
        per_expert_input_rotations=True,intermediate_hadamard=False,
        extent_alignment_slots=4,seed=0,
    )
    base=synth_layer_payloads(cfg,0)
    planes=dict(base.planes)
    for e in range(E):
        for s in range(enc["num_slots"]):
            planes[(e,s,0)]=(enc["planes"][(s,"low")],enc["planes"][(s,"high")])
    gs=base.gate_suh.clone();gs[0]=enc["suh"]
    pl=Exl3LayerPayloads(
        planes=planes,rotations=base.rotations,
        gate_suh=gs,up_suh=base.up_suh,down_svh=base.down_svh,
        sign_pattern=None,rates_fc1=None,rates_fc2=None,
    )

    args.pack_out.mkdir(parents=True,exist_ok=True)
    codes=assemble_code_rows(cfg,0,pl)
    lf=layer_filename(0)
    save_file({
        "codes":codes.contiguous(),"rotations":pl.rotations.contiguous(),
        "gate_suh":pl.gate_suh.contiguous(),"up_suh":pl.up_suh.contiguous(),
        "down_svh":pl.down_svh.contiguous(),
    },str(args.pack_out/lf),metadata=layer_metadata(cfg,0))

    md=_manifest_dict(cfg)
    md["layers"]={"0":{"file":lf,"sha256":hashlib.sha256((args.pack_out/lf).read_bytes()).hexdigest()}}
    (args.pack_out/EXL3_MANIFEST_FILENAME).write_text(json.dumps(md,indent=2,sort_keys=True)+"\n")
    rpt["checks"].append(_r("write_pack",True,pack=str(args.pack_out)))

    from b12x.moe.checkpoints.exl3 import read_exl3_manifest,read_exl3_layer
    from b12x.moe.fused_moe._impl import plan_b12x_fp4_moe_weights,prepare_b12x_fp4_moe_weights
    m=read_exl3_manifest(args.pack_out)
    g=m.geometry
    half=g.num_slots//2
    lo=read_exl3_layer(args.pack_out,m,0,first_slot=0,slot_count=half)
    dev=torch.device(args.device)
    plan=plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",source_format="exl3",activation="silu",
        params_dtype=torch.float16,num_experts=g.num_experts,
        hidden_size=K,intermediate_size=half*g.slot_channels,
        trellis_bits=args.bits,trellis_codebook=m.codebook,
        trellis_rate_granularity="uniform",
    )
    try:
        prep=prepare_b12x_fp4_moe_weights(
            plan=plan,params_dtype=torch.float16,
            exl3_layer=lo,exl3_device=dev,
        )
        val=prep.representation.value
        rpt["checks"].append(_r("prepare",True,w13_shape=list(val.w13.shape)))
    except Exception as e:
        rpt["checks"].append(_r("prepare",False,reason=str(e)))
        import traceback; rpt["traceback"]=traceback.format_exc()

    rpt["duration_seconds"]=round(time.time()-t0,3)
    out=json.dumps(rpt,indent=2,default=str)
    if args.json_out: args.json_out.write_text(out+"\n")
    print(out)
    return 0

if __name__=="__main__":
    raise SystemExit(main())