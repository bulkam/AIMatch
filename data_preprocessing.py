import numpy as np
import pandas as pd


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


def is_test_team(match, test_teams):
    return match["home_team"] in test_teams or match["away_team"] in test_teams


def onehot_encode(all_categories, category):
    onehot = np.zeros(len(all_categories))
    onehot[all_categories.index(category)] = 1
    return onehot

class Dataset:
    
    def __init__(self, 
                 path_matches = "Data/AI Match Results 150years_appended_WC2022.txt",
                 ):
        self.path = path_matches
        
        self.df = pd.DataFrame()
        self.df_labeled = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.df_labeled_testonly = pd.DataFrame()
        self.df_test_testonly = pd.DataFrame()
        
        self.all_teams = []
        self.team_to_onehot = dict()
        self.all_tournament_groups = []
        self.tournament_group_to_onehot = dict()
        
        self.X_labeled: np.array
        self.Y_labeled: np.array
        
        self.predict_values: str


    def load_raw_data(self):
        self.df = pd.read_csv(self.path)


    def preprocess_data(self,
                        drop_columns: list, 
                        strip_columns: list,
                        add_tournament_group = True):
        
        # Strip string values in columns
        for col in strip_columns:
            self.df[col] = self.df[col].apply(strip_name)
        
        # Create tournament_group column
        if add_tournament_group:
            self.df["tournament_group"] = self.df.tournament.apply(tournament_group)

        # Add home advantage column
        self.df["home_advantage"] = self.df.neutral.apply(home_advantage)

        # Drop some columns at the end
        self.df = self.df.drop(drop_columns, axis=1)


    def prepare_input_data(self, df: pd.DataFrame, predict_values = "difference", tournament_group_added = True):
        """ Performs data vectorization (categories -> onehot vectors """
        
        self.predict_values = predict_values
        
        X = np.vstack(df.home_team.apply(lambda x: self.team_to_onehot[x]))
        X = np.hstack([X, np.vstack(df.away_team.apply(lambda x: self.team_to_onehot[x]))])
        if tournament_group_added:
            X = np.hstack([X, np.vstack(df.tournament_group.apply(lambda x: self.tournament_group_to_onehot[x]))])
        X = np.hstack([X, np.vstack(df.home_advantage)])

        Y = np.hstack([np.vstack(df.home_score), np.vstack(df.away_score)])

        if self.predict_values  == "difference":
            # Apply y = [goal difference (home-away), winner_goals] instead of exact score [home_goals, away_goals]
            Y = np.hstack((np.vstack(Y[:, 0] - Y[:, 1]), np.vstack(np.max(Y, axis = 1))))

        print("X shape = ", X.shape)
        print("Y shape = ", Y.shape)

        return X, Y


    def get_input_data(self,
                        drop_columns = ["city", "country", "tournament", "neutral", "date"], 
                        strip_columns = ["home_team", "away_team", "tournament"],
                        add_tournament_group = True,
                        predict_values = "difference",
                        trainval_split = 0.7,
                        ):

        # Load and preprocess data
        self.load_raw_data()
        self.preprocess_data(drop_columns=drop_columns, strip_columns=strip_columns)
        
        # Split labeled (train + val) and test (unlabeled)
        self.df_labeled = self.df[self.df.home_score.notnull()]
        self.df_test = self.df[self.df.home_score.isnull()]

        # Teams from current cup (test)
        teams_test = set([team for team in set(self.df_test.home_team.unique().tolist() + self.df_test.away_team.unique().tolist()) if not ("group" in team.lower() or "match" in team.lower())])

        # Matches where at least one test teams is included
        self.df_labeled_testonly = self.df_labeled[self.df_labeled.apply(lambda x: is_test_team(x, teams_test), axis=1)]
        self.df_test_testonly = self.df_test[self.df_test.apply(lambda x: is_test_team(x, teams_test), axis=1)] # Without winners A type teams

        print("Number of relevant labeled matches: %s/%s" % (len(self.df_labeled_testonly), len(self.df_labeled)))

        # Data vectorization
        # Teams
        self.all_teams = sorted(list(set(list(teams_test) + [team for team in set(self.df_labeled_testonly.home_team.unique().tolist() + self.df_labeled_testonly.away_team.unique().tolist())])))
        self.team_to_onehot = dict((team, onehot_encode(self.all_teams, team)) for team in self.all_teams)
        # Tournament groups
        if add_tournament_group:
            self.all_tournament_groups = sorted(self.df.tournament_group.unique().tolist())
            self.tournament_group_to_onehot = dict((tg, onehot_encode(self.all_tournament_groups, tg)) for tg in self.all_tournament_groups)
        
        self.X_labeled, self.Y_labeled = self.prepare_input_data(self.df_labeled_testonly, predict_values=predict_values, tournament_group_added=add_tournament_group)
        X_test, _ = self.prepare_input_data(self.df_test_testonly)
        
        # Train x val split
        trainval_split_index = int(len(self.X_labeled) * trainval_split)
        X_train, Y_train = self.X_labeled[:trainval_split_index], self.Y_labeled[:trainval_split_index]
        X_val, Y_val = self.X_labeled[trainval_split_index:], self.Y_labeled[trainval_split_index:]
        
        return X_train, Y_train, X_val, Y_val, X_test
    
    
if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test = Dataset().get_input_data() # see which parameters can be modified