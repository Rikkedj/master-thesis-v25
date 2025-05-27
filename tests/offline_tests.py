
from libemg.streamers import delsys_streamer 
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter, FilePackager
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import EMGRegressor, OnlineEMGRegressor, EMGClassifier
from libemg.offline_metrics import OfflineMetrics
import re, json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

class FlutterRejectionFilter:
    """
    From source.controller.py
    """
    def __init__(self, tanh_gain=0.5, dt=0.01, integrator_enabled=False, gain=1.0, k=0.0):
        """
        Nonlinear flutter rejection filter with gain, deadband and optional integrator

        Parameters:
        -------------
        tanh_gain : float 
            The scaling factor for the tanh function. A lower tanh_gain makes the nonlinear filter more linear/gradual.
        dt : float 
            The time constant for the integrator (affects smoothness).
        integrator_enabled : bool
            If True, the filter will include an integrator to smooth the output.
        gain : float
            Gain applied to the output to scale the response.
        k : float
            New parameter, doesn't know how it make the system response
        """
        self.tanh_gain = tanh_gain
        self.dt = dt
        self.integrator_enabled = integrator_enabled
        self.state = None # Initialize the state for the integrator
        self.gain = gain
        self.k = k


    ## Updated version of filter
    def reset_integrator(self):
        """ Reset the filter state to zero. """
        if self.state is not None:
            self.state[:] = 0.0

    def filter(self, x):
        x = np.asarray(x)
        if self.state is None:
            self.state = np.zeros_like(x)
        print(f"Input: {x}")
        print(f"State: {self.state}")
        nonlinear_output = (np.abs(x)-self.k) * np.tanh(self.tanh_gain * x)
        print(f"Nonlinear Output: {nonlinear_output}")
        self.state += nonlinear_output * self.dt
        
        if self.integrator_enabled:
            return self.gain*self.state
        else:
            return self.gain*nonlinear_output
        

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
    regression_selected = True

    if regression_selected:
        data_folder = "data/regression/"
    else:
        data_folder = "data/classification/"

    data_folder = "data/regression/22-05/anders"
    json_path = os.path.join('.', data_folder, "collection_details.json")

    with open(json_path, 'r') as f:
        collection_details = json.load(f)

    num_motions = collection_details['num_motions']
    num_reps = collection_details['num_reps']
    motion_names = collection_details['classes']
    class_map = collection_details['class_map']
    
    regex_filters = [
            RegexFilter(
                left_bound = f"{data_folder}/C_",
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
    if regression_selected:
        metadata_fetchers = [
            FilePackager(RegexFilter(left_bound='animation/', right_bound='.txt', values=motion_names, description='labels'), package_function=lambda meta, data: _match_metadata_to_data(meta, data, class_map)) 
        ]
        labels_key = 'labels'
        metadata_operations = {'labels': 'last_sample'}
    else:
        metadata_fetchers = None
        labels_key = 'classes'
        metadata_operations = None

    active_classes = [2] #[0,1,2,4] # Hand open, hand close, pronation, supination
    offline_dh = OfflineDataHandler()
    offline_dh.get_data('./',regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",")
    #offline_dh.visualize()    offline_dh_c = OfflineDataHandler()
    odh_ex_rest = offline_dh.isolate_data("classes", active_classes) # Isolate the data for the active classes
    class_names = {int(c): class_map[str(c)].replace("collection_", "").replace("_", " ") for c in active_classes}
    #class_names = {class_names[c].replace("collection_", "").replace("_", " ") for c in class_names}  # Replace underscores with spaces for better readability
    
    odh_ex_rest.visualize_classes(class_names=class_names, recording_time=6) # Visualize the isolated data
    #train_odh= offline_dh.isolate_data("reps", [0,1,2])
    #test_odh = offline_dh.isolate_data("reps", [3,4])
    train_odh = odh_ex_rest.isolate_data("reps", [0,1,2])
    test_odh = odh_ex_rest.isolate_data("reps", [3,4])
    #test_odh.visualize(block=False)
    train_windows, train_metadata = train_odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT, metadata_operations=metadata_operations)
    test_windows, test_metadata = test_odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT, metadata_operations=metadata_operations)
    # Step 2: Extract features from offline data
    fe = FeatureExtractor()
    print("Extracting features")
    feature_list = fe.get_feature_list()
    desired_features = ["MAV", "ZC", "WL", "MYOP"] # The same used in Fougner 2012
    feature_list = [f for f in feature_list if f in desired_features]
    feature_list = fe.get_feature_groups()['HTD'] # Make this chosen from the GUI later
    
    training_features = fe.extract_features(feature_list, train_windows, array=True)
    test_features = fe.extract_features(feature_list, test_windows, array=True)
    test_labels = test_metadata[labels_key]
    
    visualize_features = True # Set to True to visualize the features, False to skip visualization
    if visualize_features == True:
        feat_dict = fe.extract_features(feature_list, train_windows, array=False)
        fe.visualize(feat_dict, class_names=class_names)
    # Step 3: Dataset creation
    data_set = {}
    print("Creating dataset")
    data_set['training_features'] = training_features
    data_set['training_labels'] = train_metadata[labels_key]

    filter = FlutterRejectionFilter(tanh_gain=1.5, dt=0.014, integrator_enabled=True, gain=1.1)
    test_nr = 25
    if regression_selected:
        ######## REGRESSOR ########
        # Step 4.2: Create the Regressor model
        models = ['GB'] # Make this chosen from the GUI later 'LR', 'SVM', 'RF', 'GB', 'MLP'
        results = {metric: [] for metric in ['R2', 'NRMSE', 'MAE']}
        om = OfflineMetrics()
        for model in models:
            print(f"Running {model} model")
            emg_model = EMGRegressor(model=model)
            #emg_model.install_filter(filter) #NOTE"! Made this filter 21.05 but doesn't make sense to do integration on the testdata like this, need to split it
            #emg_model.add_deadband(threshold=0.3)
            emg_model.fit(feature_dictionary=data_set)
            preds = emg_model.run(test_features)
            #filtered_preds = filter.update(preds)
            # Extract metrics for each classifier
            metrics = om.extract_offline_metrics(results.keys(), test_labels, preds)
            for metric in metrics:
                results[metric].append(metrics[metric].mean())
                
            print('Plotting decision stream. This will block the main thread once the plot is shown. Close the plot to continue.') 
            #emg_model.visualize(test_labels=test_labels, predictions=preds, save_path=f"./plots/21-05/ML-results/HTDfeatures/{model}_offline_decision_stream-filtered-{test_nr}.png", save_plot=True)
            filtered_preds = []
            for i in range(test_features.shape[0]):
                pred = preds[i]
                pred = filter.filter(pred)
                filtered_preds.append(pred)
            filtered_preds = np.array(filtered_preds)
            metrics = om.extract_offline_metrics(results.keys(), test_labels, filtered_preds)
            for metric in metrics:
                results[metric].append(metrics[metric].mean())

            plt.style.use('ggplot')
            fig, axs = plt.subplots(nrows=test_labels.shape[1], ncols=1, sharex=True, layout='constrained', figsize=(10, 7))
            fig.suptitle(f'Decision Stream for {model}', x=0.5)
            fig.supxlabel('Prediction Index')
            fig.supylabel('Model Output')

            marker_size = 5
            pred_color = 'black'
            filtered_pred_color = 'forestgreen'
            label_color = 'blue'
            x = np.arange(test_labels.shape[0])
            handles = [mpatches.Patch(color=label_color, label='True Labels'), 
                       mlines.Line2D([], [], color=pred_color, marker='o', markersize=marker_size, linestyle='None', label='Predictions'),
                       mlines.Line2D([], [], color=filtered_pred_color, linestyle='-', label='Filtered Predictions')]
            for dof_idx, ax in enumerate(axs):
                if dof_idx == 0:
                    title = "Hand Open/Close"
                    ax.text(-0.08, 1, 'Hand Open', va='center', ha='right', transform=ax.get_yaxis_transform(), fontsize=9)
                    ax.text(-0.08, -1, 'Hand Close', va='center', ha='right', transform=ax.get_yaxis_transform(), fontsize=9)
                elif dof_idx == 1:
                    title = "Pronation/Supination"
                    ax.text(-0.08, 1, 'Pronation', va='center', ha='right', transform=ax.get_yaxis_transform(), fontsize=9)
                    ax.text(-0.08, -1, 'Supination', va='center', ha='right', transform=ax.get_yaxis_transform(), fontsize=9)

                ax.set_title(f"Motor Function: {title}", fontsize=10)
                ax.set_ylim((-1.1, 1.1))
                ax.xaxis.grid(False)
                ax.fill_between(x, test_labels[:, dof_idx], alpha=0.5, color=label_color)
                ax.scatter(x, preds[:, dof_idx], color=pred_color, s=marker_size, label='Raw Preds')
                ax.scatter(x, filtered_preds[:, dof_idx], color=filtered_pred_color, s=marker_size, marker='x', label='Filtered Preds')

            fig.legend(handles=handles,
                        loc='upper right', 
                        fontsize=7, 
                        bbox_to_anchor=(0.9, 1.0))#,   bbox_transform=axs[0].transAxes)
            save_path = f"./plots/21-05/ML-results/HTDfeatures/{model}_offline_decision_stream-filtered-{test_nr}.png"
            save_path = Path(save_path)
            # Ensure save_path exists, creating it if needed
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)            
            plt.show()

        plt.style.use('ggplot')
        fig, axs = plt.subplots(ncols=len(results), layout='constrained', figsize=(10, 5))
        for metric, ax in zip(results.keys(), axs):
            print(f"Result for  metric {metric}:", results[metric])
            ax.bar(models, np.array(results[metric]) * 100, width=0.2)
            ax.set_ylabel(f"{metric} (%)")

        save_path = f"./plots/21-05/ML-results/HTDfeatures/metrics_summary-filtered-{test_nr}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.suptitle('Metrics Summary')
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    else:
        ######## CLASSIFIER ########    
        models = ['LDA', 'SVM', 'RF']
        results = {metric: [] for metric in ['CA', 'AER', 'F1', 'PREC', 'RECALL','CONF_MAT']} 
        om = OfflineMetrics()
        for model in models:
            print(f"Running {model} model")
            emg_model = EMGClassifier(model=model)
            emg_model.fit(feature_dictionary=data_set)
            preds, probs = emg_model.run(test_features)
            # Extract metrics for each classifier
            metrics = om.extract_offline_metrics(results.keys(), test_labels, preds, null_label=2) # null-label is for AER, represent the No Movement (rest) class
            for metric in metrics:
                if metric == 'CONF_MAT':
                    # Plot confusion matrix separately
                    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                    labels = [f"{i} - {motion_names[i]}" for i in range(num_motions)]
                    cm = confusion_matrix(test_labels, preds, labels=None, normalize='pred')
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                    disp.plot(xticks_rotation=45)
                    disp.ax_.set_title(f"Confusion Matrix - {model}")
                    plt.savefig(f"./plots/14-05/classification/classifier-training-protocol/confusion_matrix_{model}.png", bbox_inches='tight', dpi=300)
                    plt.show()
                results[metric].append(metrics[metric])#.mean())
            print('Plotting decision stream. This will block the main thread once the plot is shown. Close the plot to continue.') 
            #emg_model.visualize(test_labels=test_labels, predictions=preds, probabilities=probs, save_path=f"./plots/14-05/classification/classifier-training/{model}_offline_decision_stream.png", save_plot=True)

        plt.style.use('ggplot')
        fig, axs = plt.subplots(ncols=len(results)-1, layout='constrained', figsize=(10, 5))
        for metric, ax in zip(results.keys(), axs):
            if metric == 'CONF_MAT':
                # Plot confusion matrix separately
                for mat in results[metric]:
                    #om.visualize_conf_matrix(mat,labels)
                    continue
            else:
                ax.bar(models, np.array(results[metric]) * 100, width=0.2)
                ax.set_ylabel(f"{metric} (%)")

        fig.suptitle('Metrics Summary')
        fig.savefig(f"./plots/14-05/classification/classifier-training-protocol/metrics_summary.png")
        plt.show()

        om.extract_offline_metrics(['CONF_MAT'],test_labels, preds)
    

    
