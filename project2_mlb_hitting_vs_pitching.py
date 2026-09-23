# MLB Hitting vs Pitching Project
# See notebook/report for the full analysis.

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/corbtastik/lahman-baseball-db/master/Teams.csv"

df = pd.read_csv(DATA_URL)
df = df[df["yearID"].between(2000, 2025)].copy()

# Team-season features
# Target: wins (W)
df["runs_per_game"] = df["R"] / df["G"]
df["runs_allowed_per_game"] = df["RA"] / df["G"]
df["home_runs_per_game"] = df["HR"] / df["G"]
df["obp"] = (df["H"] + df["BB"] + df["HBP"]) / (df["AB"] + df["BB"] + df["HBP"] + df["SF"])
df["slg"] = (df["H"] + df["2B"] + 2*df["3B"] + 3*df["HR"]) / df["AB"]
df["so_per_9"] = df["SOA"] / (df["IPouts"] / 3) * 9

print(df[["yearID", "name", "W", "runs_per_game", "runs_allowed_per_game", "obp", "slg", "ERA", "so_per_9"]].head())
