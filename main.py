from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import networkx as nx
import io
import json
from typing import Optional

app = FastAPI(title="SNA Influence Mapper API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "SNA Influence Mapper API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    name_col: str = Form(...),
    dept_col: str = Form(...),
    desig_col: Optional[str] = Form("skip"),
    top_n: int = Form(5),
    unit_definitions: str = Form("[]"),
    fallback_color: str = Form("#9E9E9E"),
    org_name: str = Form("My Organisation"),
    skip_rows: int = Form(0),
):
    content = await file.read()
    if file.filename.endswith(".csv"):
        df_raw = pd.read_csv(io.BytesIO(content))
    else:
        df_raw = pd.read_excel(io.BytesIO(content))

    if skip_rows > 0:
        df_raw = df_raw.iloc[skip_rows:].reset_index(drop=True)

    unit_defs = json.loads(unit_definitions)

    def map_unit(raw):
        for u in unit_defs:
            if u["keyword"].lower() in str(raw).lower():
                return u["name"]
        return "Other"

    def get_color(unit_name):
        for u in unit_defs:
            if u["name"] == unit_name:
                return u["color"]
        return fallback_color

    def ct(x):
        return "Unknown" if pd.isna(x) else str(x).replace("\n", " ").strip()

    df_name = df_raw[name_col].apply(ct)
    df_dept = df_raw[dept_col].apply(ct).apply(map_unit)
    if desig_col != "skip" and desig_col in df_raw.columns:
        df_desig = df_raw[desig_col].apply(ct)
    else:
        df_desig = pd.Series(["Not specified"] * len(df_raw))

    df = pd.DataFrame({"name": df_name, "dept": df_dept, "desig": df_desig})
    df = df[(df["name"] != "Unknown") & (df["name"].str.len() >= 2)]

    if len(df) < 3:
        return {"error": "Not enough valid data. Check column mapping."}

    G = nx.Graph()
    for dept, people in df.groupby("dept")["name"].apply(list).items():
        if len(people) >= 2:
            for i in range(len(people)):
                for j in range(i + 1, len(people)):
                    G.add_edge(people[i], people[j], weight=2)

    if desig_col != "skip" and desig_col in df_raw.columns:
        for desig, people in df.groupby("desig")["name"].apply(list).items():
            if 2 <= len(people) <= 10:
                for i in range(len(people)):
                    for j in range(i + 1, len(people)):
                        di = df[df["name"] == people[i]]["dept"].values
                        dj = df[df["name"] == people[j]]["dept"].values
                        if len(di) > 0 and len(dj) > 0 and di[0] != dj[0]:
                            if not G.has_edge(people[i], people[j]):
                                G.add_edge(people[i], people[j], weight=1)

    if G.number_of_nodes() == 0:
        return {"error": "Could not build network. Check your data."}

    dc = nx.degree_centrality(G)
    bc = nx.betweenness_centrality(G, normalized=True)
    cc = nx.closeness_centrality(G)
    pr = nx.pagerank(G, weight="weight")

    results = []
    for person in df["name"].unique():
        if person in G:
            score = (0.30 * dc.get(person, 0)
                     + 0.40 * bc.get(person, 0)
                     + 0.20 * cc.get(person, 0)
                     + 0.10 * pr.get(person, 0))
            p = df[df["name"] == person].iloc[0]
            results.append({
                "Name": person,
                "Designation": p["desig"],
                "Unit": p["dept"],
                "Color": get_color(p["dept"]),
                "Connections": G.degree(person),
                "Betweenness": round(bc.get(person, 0), 4),
                "Closeness": round(cc.get(person, 0), 4),
                "PageRank": round(pr.get(person, 0), 4),
                "InfluenceScore": score,
            })

    rdf = sorted(results, key=lambda x: x["InfluenceScore"], reverse=True)
    mx = rdf[0]["InfluenceScore"] if rdf else 1
    for r in rdf:
        r["Influence"] = round(r["InfluenceScore"] / mx * 100, 1) if mx > 0 else 0

    edges = [{"source": u, "target": v} for u, v in G.edges()]

    unit_counts = df["dept"].value_counts().to_dict()

    return {
        "org_name": org_name,
        "total_people": len(df),
        "total_connections": G.number_of_edges(),
        "total_units": df["dept"].nunique(),
        "density": round(nx.density(G), 4),
        "top_influencers": rdf[:top_n],
        "all_rankings": rdf,
        "edges": edges[:500],
        "unit_counts": unit_counts,
    }


@app.post("/columns")
async def get_columns(file: UploadFile = File(...)):
    content = await file.read()
    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content), nrows=0)
    else:
        df = pd.read_excel(io.BytesIO(content), nrows=0)
    return {"columns": list(df.columns)}
