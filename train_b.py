import numpy as np
import keras
from keras.models import Model, save_model
from keras.optimizers import SGD
from keras.layers import Input, Dense
from keras.callbacks import Callback
import data_preprocessing as prep
import logging

logging.basicConfig(filename="train.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running Urban Planning")

logger = logging.getLogger('urbanGUI')


def score_sample(y_pred, y_true, label_weights):
    """ 
    Scoring according to the AI match rules
    y_pred = predictions
    y_true = ground truth 
    """
    
    y = np.around(y_pred / label_weights)
    y_gt = np.around(y_true / label_weights)
    #print(y, y_pred, y_true, y_gt)

    if y[0] == y_gt[0] and y[1] == y_gt[1]:
        return 4
    
    if y[0] == y_gt[0] and not y[0] == 0:
        return 3

    if (y[0] >= 0 and y_gt[0] >= 0) or (y[0] < 0 and y_gt[0] < 0):
        return 2   

    return 0


def score(Y, Y_gt, label_weights=[1, 1]):
    return np.sum([score_sample(Y[i], Y_gt[i], np.array(label_weights)) for i in range(len(Y))])


def prediction_to_goals(y_weighted, label_weights):
    """ Encoding back to the exact result 
    [goal_diff, home_goals] -> [home_goals, away_goals]
    """
    y = y_weighted / np.array(label_weights)
    return np.hstack([y[1], y[1] - y[0]])


def predictions_to_goals(Y, label_weights):
    """ Encoding back to the exact result for whole data set
    [goal_diff, home_goals] -> [home_goals, away_goals]
    """
    return np.vstack([prediction_to_goals(Y[i], label_weights) for i in range(len(Y))])


def show_predictions(dataset, X, Y, Y_pred, indexes, label_weights):
    for i in indexes:
        x = X[i]
        home_team_part = x[:len(dataset.all_teams)]
        away_team_part = x[len(dataset.all_teams):2*len(dataset.all_teams)]
        home_team = dataset.all_teams[np.where(home_team_part == 1)[0][0]]
        away_team = dataset.all_teams[np.where(away_team_part == 1)[0][0]]
        print(home_team, " x ", away_team, ": ", prediction_to_goals(Y_pred[i], label_weights), "-", Y[i], " ...................  output (weighted): ", Y_pred[i], Y[i], "   original: ", Y_pred[i]/label_weights, Y[i]/label_weights)
        logging.info([home_team, " x ", away_team, ": ", prediction_to_goals(Y_pred[i], label_weights), "-", Y[i], " ...................  output (weighted): ", Y_pred[i], Y[i], "   original: ", Y_pred[i]/label_weights, Y[i]/label_weights])


class Scorer(Callback):
    def __init__(self, X, Y, label_weights):
        self.X_val, self.Y_val = X, Y
        self.label_weights = label_weights
        
    def on_epoch_end(self, batch, logs={}):
        Y_pred = np.hstack(self.model.predict(self.X_val))
        print("X_val score = ", score(Y_pred, self.Y_val, label_weights=self.label_weights))
        logging.info(["X_val score = ", score(Y_pred, self.Y_val, label_weights=self.label_weights)])
        return
    

if __name__ == "__main__":
    
    dataset = prep.Dataset()
    X_train, Y_train, X_val, Y_val, X_test, sample_weights_train, sample_weights_val = dataset.get_input_data(label_weights=[1, 1], sample_weights_degree=2)

    # Reference values
    # = total score for validation data if results are hard-coded and all same without any prediction
    # all models should overcome those values
    print("    Reference values:")
    max_score = 4 * len(Y_val)
    ref_score_1 = score(np.zeros(Y_val.shape) * dataset.label_weights, Y_val, label_weights=dataset.label_weights) # 0:0
    print("0:0", ref_score_1, "/", max_score, " - %s points per match" % (np.round(ref_score_1/len(Y_val), 2)))
    ref_score_2 = score(np.ones(Y_val.shape) * dataset.label_weights, Y_val, label_weights=dataset.label_weights) # 1:0
    print("1:0", ref_score_2, "/", max_score, " - %s points per match" % (np.round(ref_score_2/len(Y_val), 2)))
    Y_pred = np.ones(Y_val.shape) * dataset.label_weights # 1:1
    Y_pred[:, 0] = 0
    ref_score_3 = score(Y_pred, Y_val, label_weights=dataset.label_weights) 
    print("1:1", ref_score_3, "/", max_score, " - %s points per match" % (np.round(ref_score_3/len(Y_val), 2)))
    Y_pred = np.ones(Y_val.shape) * dataset.label_weights # 0:1
    Y_pred[:, 0] = -1
    ref_score_3 = score(Y_pred, Y_val, label_weights=dataset.label_weights) 
    print("0:1", ref_score_3, "/", max_score, " - %s points per match" % (np.round(ref_score_3/len(Y_val), 2)))
    Y_pred = np.ones(Y_val.shape) # 2:1
    Y_pred[:, 1] = 2
    ref_score_3 = score(Y_pred * dataset.label_weights, Y_val, label_weights=dataset.label_weights) 
    print("2:1", ref_score_3, "/", max_score, " - %s points per match" % (np.round(ref_score_3/len(Y_val), 2)))
    Y_pred = 2 * np.ones(Y_val.shape) # 2:0
    ref_score_3 = score(Y_pred * dataset.label_weights, Y_val, label_weights=dataset.label_weights) 
    print("2:0", ref_score_3, "/", max_score, " - %s points per match" % (np.round(ref_score_3/len(Y_val), 2)))
        

    model_input = Input(shape=(X_train.shape[1],)) 
    # First branch
    a_dense_1 = Dense(128, activation = "relu")(model_input)
    a_dense_2 = Dense(32, activation = "relu")(a_dense_1)
    a_dense_3 = Dense(8, activation = "relu")(a_dense_2)
    a_dense_4 = Dense(1, name = "goal_diff", activation = "linear")(a_dense_2)
    # Second branch
    b_dense_1 = Dense(128, activation = "relu")(model_input)
    b_dense_2 = Dense(32, activation = "relu")(b_dense_1)
    b_dense_3 = Dense(8, activation = "relu")(b_dense_2)
    b_dense_4 = Dense(1, name = "winner_goals", activation = "relu")(b_dense_1)

    model = Model(model_input, outputs=[a_dense_4, b_dense_4])

    optimizer = SGD(lr=0.02)
    model.compile(optimizer=optimizer,loss={'goal_diff': 'mse', 'winner_goals': 'mae'}, metrics={'goal_diff': 'mse', 'winner_goals': 'mae'})

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5, batch_size=16, callbacks=[Scorer(X_val, Y_val, dataset.label_weights)], shuffle=True)


    Y_test_pred = np.hstack(model.predict(X_test))
    print(Y_test_pred)
    Y_val_pred = np.hstack(model.predict(X_val))
    
    show_predicted_indexes = [i for i in range(len(X_test))]

    #print("Val:")
    #show_predictions(X_val, Y_val, Y_val_pred, show_predicted_indexes, dataset.label_weights)
    print("Test:")
    show_predictions(X_test, np.zeros(Y_test_pred.shape), Y_test_pred, show_predicted_indexes, dataset.label_weights)