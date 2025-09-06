
import json
from typing import List, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pulp

# Cargar config por defecto
with open("model_config.json","r") as f:
    DEFAULT = json.load(f)

class Plant(BaseModel):
    plant: str
    capacity_kg: float
    prod_cost_per_kg: float
    yield_: float = 1.0

class DC(BaseModel):
    dc: str
    demand_kg: float

class Transport(BaseModel):
    plant: str
    dc: str
    transp_cost_per_kg: float

class OptimizeRequest(BaseModel):
    plants: Optional[List[Plant]] = None
    dcs: Optional[List[DC]] = None
    transport: Optional[List[Transport]] = None
    penalty_stockout: Optional[float] = None

app = FastAPI(title="Snack Production & Distribution Optimizer")

def to_df(lst, cols):
    if lst is None: return None
    df = pd.DataFrame([x.dict() for x in lst])
    # Renombrar yield_->yield para consistencia interna
    if "yield_" in df.columns:
        df = df.rename(columns={"yield_":"yield"})
    return df[cols]

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    plants = to_df(req.plants, ["plant","capacity_kg","prod_cost_per_kg","yield"]) \
             if req.plants is not None else pd.DataFrame(DEFAULT["plants"])
    dcs = to_df(req.dcs, ["dc","demand_kg"]) \
          if req.dcs is not None else pd.DataFrame(DEFAULT["dcs"])
    transport = to_df(req.transport, ["plant","dc","transp_cost_per_kg"]) \
                if req.transport is not None else pd.DataFrame(DEFAULT["transport"])
    penalty = req.penalty_stockout if req.penalty_stockout is not None else DEFAULT["penalty_stockout"]

    P = plants["plant"].tolist()
    D = dcs["dc"].tolist()

    cap = dict(zip(plants["plant"], plants["capacity_kg"]*plants["yield"]))
    c_prod = dict(zip(plants["plant"], plants["prod_cost_per_kg"]))
    c_transp = {(r.plant, r.dc): r.transp_cost_per_kg for r in transport.itertuples(index=False)}
    demand = dict(zip(dcs["dc"], dcs["demand_kg"]))

    m = pulp.LpProblem("Snack_Prod_Distrib", pulp.LpMinimize)

    produce = pulp.LpVariable.dicts("produce", P, lowBound=0)
    ship = pulp.LpVariable.dicts("ship", [(p,d) for p in P for d in D], lowBound=0)
    short = pulp.LpVariable.dicts("short", D, lowBound=0)

    m += (
        pulp.lpSum(c_prod[p]*produce[p] for p in P) +
        pulp.lpSum(c_transp.get((p,d), 1e6)*ship[(p,d)] for p in P for d in D) +
        penalty*pulp.lpSum(short[d] for d in D)
    )

    for p in P:
        m += produce[p] <= cap[p]
        m += produce[p] >= pulp.lpSum(ship[(p,d)] for d in D)

    for d in D:
        m += pulp.lpSum(ship[(p,d)] for p in P) + short[d] == demand[d]

    m.solve(pulp.PULP_CBC_CMD(msg=False))

    res = {
        "status": pulp.LpStatus[m.status],
        "total_cost": pulp.value(m.objective),
        "produce": [{ "plant": p, "produce_kg": produce[p].value() } for p in P],
        "ship": [{ "plant": p, "dc": d, "ship_kg": ship[(p,d)].value() } for p in P for d in D],
        "short": [{ "dc": d, "short_kg": short[d].value() } for d in D]
    }
    # Filtrar pequeñas tolerancias numéricas
    res["ship"] = [r for r in res["ship"] if r["ship_kg"] and r["ship_kg"] > 1e-6]
    return res
