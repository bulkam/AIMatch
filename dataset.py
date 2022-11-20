import numpy as np
import tensorflow as tf
import keras
import csv

class Dataset(tf.keras.utils.Sequence):
    def __init__(self, source_path):
        super().__init__()
        self.source_path = source_path
        self.source_data = []
        self.teams = []
        self.team_numbers = dict()
        self.team_count = 0
        self.headers = [
                        "home_team",
                        "away_team",
                        "home_score",
                        "away_score"
                    ]

        with open(self.source_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                self.source_data.append(dict())
                for key, value in row.items():
                    if key in self.headers:
                        self.source_data[i][key] = value
                    if key in ["home_team", "away_team"]:
                        if row[key] not in self.teams:
                            self.team_numbers[value] = self.team_count
                            self.teams.append(value)
                            self.team_count += 1

        self.cleaned_data = []
        for row in self.source_data:
            t = False
            for header in self.headers:
                if row[header] == '':
                    t = True

            if not t:
                self.cleaned_data.append(row)
        print(self.team_count)

    def input_d(self):
        return 2*self.team_count

    def __len__(self):
        return len(self.cleaned_data)

    def __getitem__(self, idx):
        data = self.cleaned_data[idx]
        home_team = []
        away_team = []
        for i in range(self.team_count):
            home_team.append(0)
            away_team.append(0)
            if i == self.team_numbers[data["home_team"]]:
                home_team[i] = 1
            if i == self.team_numbers[data["away_team"]]:
                away_team[i] = 1

        match = np.array([np.append(home_team, away_team)])

        score = np.array([int(data["home_score"]), int(data["away_score"])])

        return match, score
