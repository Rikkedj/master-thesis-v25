
from libemg.streamers import delsys_streamer 
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter, FilePackager
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import EMGRegressor, OnlineEMGRegressor
from libemg.offline_metrics import OfflineMetrics
import re, json, os
import numpy as np
import matplotlib.pyplot as plt

def _match_metadata_to_data(metadata_file: str, data_file: str, class_map: dict) -> bool:
            """
            Ensures the correct animation metadata file is matched with the correct EMG data file.

            Args:
                metadata_file (str): Metadata file path (e.g., "animation/collection_hand_open_close.txt").
                data_file (str): EMG data file path (e.g., "data/regression/C_0_R_01_emg.csv").
                class_map (dict): Dictionary mapping class index (str) to motion filenames.

            Returns:
                bool: True if the metadata file corresponds to the class of the data file.
            """
            # Extract class index from data filename (C_{k}_R pattern)
            match = re.search(r"C_(\d+)_R", data_file)
            if not match:
                return False  # No valid class index found

            class_index = match.group(1)  # Extract class index as a string

            # Find the expected metadata file from class_map
            expected_metadata = class_map.get(class_index)
            if not expected_metadata:
                return False  # No matching motion found

            # Construct the expected metadata filename
            expected_metadata_file = f"animation/{expected_metadata}.txt"

            return metadata_file == expected_metadata_file

if __name__ == '__main__':
    # need to have the same amount of channels that you had when training the model
    #_, sm = delsys_streamer(channel_list=[0,4,8,9,13]) # returns streamer and the shared memory object, need to give in the active channels number -1, so 2 is sensor 3
    #odh = OnlineDataHandler(sm) # Offline datahandler on shared memory
    
    ############ Set up regressor ########
    # Step 2.1: Parse offline training data
    WINDOW_SIZE = 200       # 100 ms * 2000Hz = 200 samples
    WINDOW_INCREMENT = 100   # 50 ms * 2000Hz = 10 samples

    data_folder = "data/regression/"
    json_path = os.path.join(data_folder, "collection_details.json")

    with open(json_path, 'r') as f:
        collection_details = json.load(f)

    num_motions = collection_details['num_motions']
    num_reps = collection_details['num_reps']
    motion_names = collection_details['classes']
    class_map = collection_details['class_map']
    
    regex_filters = [
        RegexFilter(
            left_bound = os.path.join(data_folder, "C_"),
            right_bound = "_R",
            values = [str(i) for i in range(num_motions)],
            description = 'classes'
        ),
        RegexFilter(
            left_bound = "R_", 
            right_bound="_emg.csv", 
            values = [str(i) for i in range(num_reps)], 
            description='reps'
        )
    ]
    metadata_fetchers = [
        FilePackager(RegexFilter(left_bound='animation/', right_bound='.txt', values=motion_names, description='labels'), package_function=lambda meta, data: _match_metadata_to_data(meta, data, class_map)) 
    ]
    labels_key = 'labels'
    metadata_operations = {'labels': 'last_sample'}

    offline_dh = OfflineDataHandler()
    offline_dh.get_data('./', regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",")
    #offline_dh.visualize()
    
    train_odh= offline_dh.isolate_data("reps", [0,1,2])
    test_odh = offline_dh.isolate_data("reps", [3,4])
    test_odh.visualize(block=False)
    train_windows, train_metadata = train_odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT, metadata_operations=metadata_operations)
    test_windows, test_metadata = test_odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT, metadata_operations=metadata_operations)
    # Step 2: Extract features from offline data
    fe = FeatureExtractor()
    print("Extracting features")
    feature_list = fe.get_feature_list()
    desired_features = ["MAV", "ZC", "WL", "MYOP"]
    feature_list = [f for f in feature_list if f in desired_features]
    #feature_list = fe.get_feature_groups()['HTD'] # Make this chosen from the GUI later
    
    training_features = fe.extract_features(feature_list, train_windows, array=True)
    test_features = fe.extract_features(feature_list, test_windows, array=True)
    test_labels = test_metadata[labels_key]

    # Step 3: Dataset creation
    data_set = {}
    print("Creating dataset")
    data_set['training_features'] = training_features
    data_set['training_labels'] = train_metadata[labels_key]

    # Step 4: Create the EMG model
    
    models = ['LR', 'SVM', 'RF', 'GB', 'MLP'] # Make this chosen from the GUI later
    results = {metric: [] for metric in ['R2', 'NRMSE', 'MAE']}
    om = OfflineMetrics()

    for model in models:
        print(f"Running {model} model")
        emg_model = EMGRegressor(model=model)
        emg_model.fit(feature_dictionary=data_set)
        preds = emg_model.run(test_features)

        # Extract metrics for each classifier
        metrics = om.extract_offline_metrics(results.keys(), test_labels, preds)
        for metric in metrics:
            results[metric].append(metrics[metric].mean())
        #emg_model.visualize(test_labels=test_labels, predictions=preds)
        print('Plotting decision stream. This will block the main thread once the plot is shown. Close the plot to continue.') 
        emg_model.visualize(test_labels=test_labels, predictions=preds)

    plt.style.use('ggplot')
    fig, axs = plt.subplots(ncols=len(results), layout='constrained', figsize=(10, 5))
    for metric, ax in zip(results.keys(), axs):
        ax.bar(models, np.array(results[metric]) * 100, width=0.2)
        ax.set_ylabel(f"{metric} (%)")

    fig.suptitle('Metrics Summary')
    plt.show()
