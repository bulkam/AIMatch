import numpy as np
import pandas as pd


def load_raw_data(path_matches = "Data/AI Match Results 150years_appended_WC2022.txt"):
    return pd.read_csv(path_matches)


def strip_name(s: str):
    return s.strip()


def tournament_group(tournament: str):
    t = tournament.lower().replace(" ","")

    if t == "friendly":
        return "Friendly"

    if t.endswith("worldcup") or t.endswith("worldcup2022"):
        return "World Cup"
    
    if t.endswith("euro") or t.endswith("copaamérica") or t.endswith("africancupofnations") or t.endswith("asiancup"):
        return "Continental Cup"

    if t.endswith("worldcupqualification") or t.endswith("worldcup2022qualification"):
        return "World Cup Qualification"
    
    if t.endswith("euroqualification") or t.endswith("copaaméricaqualification") or t.endswith("africancupofnationsqualification") or t.endswith("asiancupqualification"):
        return "Continental Cup Qualification"
    
    return "Others"


def home_advantage(neutral):
    return 0 if neutral else 1


def preprocess_data(df: pd.DataFrame, 
                    drop_columns: list, 
                    strip_columns: list,
                    add_tournament_group = True):
    
    # Strip string values in columns
    for col in strip_columns:
        df[col] = df[col].apply(strip_name)
    
    # Create tournament_group column
    if add_tournament_group:
        df["tournament_group"] = df.tournament.apply(tournament_group)

    # Add home advantage column
    df["home_advantage"] = df.neutral.apply(home_advantage)

    # Drop some columns at the end
    df = df.drop(drop_columns, axis=1)
    
    return df

    

