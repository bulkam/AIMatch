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


def is_test_team(match, teams):
    return match["home_team"] in teams or match["away_team"] in teams


def onehot_encode(all_categories, category):
    onehot = np.zeros(len(all_categories))
    onehot[all_categories.index(category)] = 1
    return onehot


def get_input_data(df: pd.DataFrame, 
                    drop_columns: list, 
                    strip_columns: list,
                    add_tournament_group = True,
                    ):

    # Load and preprocess data
    df = load_raw_data()
    df = preprocess_data(df, 
            drop_columns=["city", "country", "tournament", "neutral", "date"], 
            strip_columns=["home_team", "away_team", "tournament"])
    
    # Split labeled (train + val) and test (unlabeled)
    df_labeled = df[df.home_score.notnull()]
    df_test = df[df.home_score.isnull()]

    # Teams from current cup (test)
    teams_test = set([team for team in set(df_test.home_team.unique().tolist() + df_test.away_team.unique().tolist()) if not ("group" in team.lower() or "match" in team.lower())])

    # Matches where at least one test teams is included
    df_labeled_testonly = df_labeled[df_labeled.apply(lambda x: is_test_team(x, teams_test), axis=1)]
    df_test_testonly = df_test[df_test.apply(lambda x: is_test_team(x, teams_test), axis=1)] # Without winners A type teams

    print("Number of relevant labeled matches: %s/%s" % (len(df_labeled_testonly), len(df_labeled)))

    # Data vectorization
    # Teams
    all_teams = sorted(list(set(list(teams_test) + [team for team in set(df_labeled_testonly.home_team.unique().tolist() + df_labeled_testonly.away_team.unique().tolist())])))
    team_to_onehot = dict((team, onehot_encode(all_teams, team)) for team in all_teams)
    # Tournament groups
    all_tournament_groups = sorted(df.tournament_group.unique().tolist())
    tournament_group_to_onehot = dict((tg, onehot_encode(all_tournament_groups, tg)) for tg in all_tournament_groups)



    

