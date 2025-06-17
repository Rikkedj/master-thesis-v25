'''
This code is taken from LibEMG IsoFitts Showcase, https://github.com/LibEMG/LibEMG_Isofitts_Showcase.git
'''
from pathlib import Path

import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from os import walk
from libemg.emg_predictor import EMGClassifier, EMGRegressor
from libemg.feature_extractor import FeatureExtractor
from libemg.data_handler import OfflineDataHandler, RegexFilter, FilePackager
from libemg.offline_metrics import OfflineMetrics

def evaluate_offline_data():
    WINDOW_SIZE = 40
    WINDOW_INCREMENT = 20
    training_data_folder = 'data/regression/separated-motions/s1-04-06-unconstrained'
    media_folder = 'animations/saw_tooth'
    offline_metrics = {
        'model': [],
        'predictor_type': [],
        'metrics': [],
    }

    regex_filters = [
        RegexFilter(left_bound = "classification/C_", right_bound="_R", values = ["0","1","2","3","4"], description='classes'),
        RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = ["0", "1", "2"], description='reps'),
    ]

    clf_odh = OfflineDataHandler()
    clf_odh.get_data('data/', regex_filters, delimiter=",")

    regex_filters = [
        RegexFilter(left_bound='data/regression/C_0_R_', right_bound='_emg.csv', values=['0', '1', '2'], description='reps')
    ]
    metadata_fetchers = [
        FilePackager(RegexFilter(left_bound='animation/', right_bound='.txt', values=['collection'], description='labels'), package_function=lambda x, y: True)
    ]
    reg_odh = OfflineDataHandler()
    reg_odh.get_data('./', regex_filters, metadata_fetchers=metadata_fetchers, delimiter=',')


    fe = FeatureExtractor()

    for odh in [clf_odh, reg_odh]:
        is_regression = odh == reg_odh
        if is_regression:
            metadata_operations = {'labels': 'last_sample'}
            metrics = ['MAE', 'MSE', 'NRMSE']
            labels_key = 'labels'
            models = ['LR', 'RF']
        else:
            metadata_operations = None
            metrics = ['CA', 'AER', 'INS', 'CONF_MAT']
            labels_key = 'classes'
            models = ['LDA', 'SVM', 'KNN', 'RF']
        train_odh = odh.isolate_data(key="reps", values=[0,1])
        train_windows, train_metadata = train_odh.parse_windows(WINDOW_SIZE,WINDOW_INCREMENT, metadata_operations=metadata_operations)
        test_odh = odh.isolate_data(key="reps", values=[2])
        test_windows, test_metadata = test_odh.parse_windows(WINDOW_SIZE,WINDOW_INCREMENT, metadata_operations=metadata_operations)

        data_set = {}
        data_set['testing_features'] = fe.extract_feature_group('HTD', test_windows)
        data_set['training_features'] = fe.extract_feature_group('HTD', train_windows)
        data_set['testing_labels'] = test_metadata[labels_key]
        data_set['training_labels'] = train_metadata[labels_key]

        om = OfflineMetrics()
        # Normal Case - Test all different models
        for model in models:
            if is_regression:
                emg_predictor = EMGRegressor(model)
                emg_predictor.fit(data_set.copy())
                preds = emg_predictor.run(data_set['testing_features'])
                predictor_type = 'regression'
            else:
                emg_predictor = EMGClassifier(model)
                emg_predictor.fit(data_set.copy())
                preds, _ = emg_predictor.run(data_set['testing_features'])
                predictor_type = 'classification'
            out_metrics = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, 2)
            offline_metrics['model'].append(model)
            offline_metrics['predictor_type'].append(predictor_type)
            offline_metrics['metrics'].append(out_metrics)
    return offline_metrics

def read_pickle(location):
    with open(location, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_overshoots(data):
    overshoots = 0
    trials = np.unique(data['trial_number'])
    for t in trials:
        t_idxs = np.where(data['trial_number'] == t)[0]
        cursor_locs = np.array(data['cursor_position'])[t_idxs]
        targets = np.array(data['goal_circle'])[t_idxs]
        in_bounds = [in_circle(cursor_locs[i], targets[i]) for i in range(0,len(cursor_locs))]
        for i in range(1,len(in_bounds)):
            if in_bounds[i-1] == True and in_bounds[i] == False:
                overshoots+=1 
    return overshoots

def in_circle(cursor, circle):
    return math.sqrt((circle[0] - cursor[0])**2 + (circle[1] - cursor[1])**2) < circle[2]/2 + cursor[2]/2

def calculate_efficiency(data):
    efficiency = []
    trials = np.unique(data['trial_number'])
    for t in trials:
        t_idxs = np.where(data['trial_number'] == t)[0]
        distance_travelled = np.sum([math.dist(data['cursor_position'][t_idxs[i]][0:2], data['cursor_position'][t_idxs[i-1]][0:2]) for i in range(1,len(t_idxs))])
        fastest_path = math.dist((data['cursor_position'][t_idxs[0]])[0:2], (data['goal_circle'][t_idxs[0]])[0:2])
        efficiency.append(fastest_path/distance_travelled)
    return np.mean(efficiency)

def calculate_throughput(data):
    throughput = []
    trials = np.unique(data['trial_number'])
    for t in trials:
        t_idxs = np.where(data['trial_number'] == t)[0]
        distance = math.dist((data['cursor_position'][t_idxs[0]])[0:2], (data['goal_circle'][t_idxs[0]])[0:2])
        width = (data['goal_circle'][t_idxs[0]])[2]
        id = math.log2(distance/width + 1) 
        time = data['global_clock'][t_idxs[-1]] - data['global_clock'][t_idxs[0]]
        throughput.append(id/time)
    return np.mean(throughput)

def extract_fitts_metrics(data):
    fitts_results = {
        'overshoots': [],
        'throughput': [],
        'efficiency': [],
    }
    fitts_results['overshoots'] = calculate_overshoots(data)
    fitts_results['efficiency'] = calculate_efficiency(data)
    fitts_results['throughput'] = calculate_throughput(data)
    return fitts_results


def evaluate_fitts_data():
    path = 'results/'
    filenames = next(walk(path), (None, None, []))[2]
    filenames = [filename for filename in filenames if filename.endswith('.pkl')]
    fitts_metrics = {
        'model': [],
        'metrics': [],
    }
    
    for file in filenames:
        data = read_pickle(path + file)
        fitts_metrics['model'].append(Path(file).stem)
        fitts_metrics['metrics'].append(extract_fitts_metrics(data))

    return fitts_metrics

if __name__ == "__main__":
    offline_metrics = evaluate_offline_data()
    fitts_metrics = evaluate_fitts_data()

    # Plot bar chart for each model - lets look at CA, AER and INS for each classifier and MAE, MSE, and NRMSE for each regressor
    clf_mets = ['CA', 'AER', 'INS']
    reg_mets = ['MAE', 'MSE', 'NRMSE']
    f_mets = ['throughput', 'efficiency', 'overshoots']
    fig, axs = plt.subplots(len(clf_mets), 3)
    for i in range(0, 3):
        # Plot CA, AER and INS
        clf_mask = [idx for idx, x in enumerate(offline_metrics['predictor_type']) if x == 'classification']
        clf_x = [offline_metrics['model'][idx] for idx in clf_mask]
        clf_y = [(offline_metrics['metrics'][y])[clf_mets[i]] for y in clf_mask]
        axs[i, 0].bar(clf_x,clf_y)
        axs[i, 0].set_ylabel(clf_mets[i])

        # Plot regression metrics
        reg_mask = [idx for idx, x in enumerate(offline_metrics['predictor_type']) if x == 'regression']
        reg_x = [offline_metrics['model'][idx] for idx in reg_mask]
        reg_y = [(offline_metrics['metrics'][y])[reg_mets[i]].mean() for y in reg_mask]
        axs[i, 1].bar(reg_x,reg_y)
        axs[i, 1].set_ylabel(reg_mets[i])
        
        # Plot throughput, efficiency and overshoots 
        fitts_x = [x for x in fitts_metrics['model']]
        fitts_y = [(fitts_metrics['metrics'][y])[f_mets[i]] for y in range(0, len(fitts_metrics['model']))]
        axs[i, 2].bar(fitts_x, fitts_y)
        axs[i, 2].set_ylabel(f_mets[i].title())
    
    axs[0, 0].set_title('Classification Metrics')
    axs[0, 1].set_title('Regression Metrics')
    axs[0, 2].set_title('Usability Metrics')
    plt.tight_layout()
    plt.show()

    print(offline_metrics)
    print(fitts_metrics)