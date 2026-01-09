"""
Dataset pipeline for NEURON-reduce simulation outputs.

Reads pkl files produced by example/example.py (keys: voltage, exInputSpikeTimes,
inhInputSpikeTimes, outputSpikeTimes), converts them into a dataset format similar
to IF_dataset_pipeline.py, and organizes train/valid/test splits.
"""

import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ReduceDatasetPipeline:
    """
    Dataset pipeline for NEURON-reduce simulations.
    Processes pkl files and organizes them into train/validation/test sets.
    """

    def __init__(self, root_folder_path, output_dir, train_dir, valid_dir, test_dir):
        self.root_folder_path = Path(root_folder_path)

        self.output_dir = Path(output_dir)
        self.train_dir = Path(train_dir)
        self.valid_dir = Path(valid_dir)
        self.test_dir = Path(test_dir)

        self._create_directories()

    def _create_directories(self):
        for d in [self.output_dir, self.train_dir, self.valid_dir, self.test_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def load_trial_data(self, trial_id):
        """
        Load one NEURON-reduce pkl and convert to common dict format.
        """
        try:
            pickle_path = self.root_folder_path / f"trial_{trial_id}.pkl"
            with open(pickle_path, "rb") as f:
                sim_dict = pickle.load(f)

            voltage = sim_dict["voltage"]
            ex_input_raster = sim_dict["exInputSpikeTimes"]
            inh_input_raster = sim_dict["inhInputSpikeTimes"]
            output_spikes = sim_dict["outputSpikeTimes"]

            # infer dt from voltage length and time window (raster time axis)
            n_time_steps = voltage.shape[0]
            dt = 1.0  # ms, default fallback
            if ex_input_raster.shape[1] > 1:
                # assume raster uses same dt as voltage
                dt = float(sim_dict.get("dt", 1.0))
                if "dt" not in sim_dict:
                    dt = 1.0  # best effort

            # downsample voltage to 1 ms resolution (mean)
            ratio = max(1, int(round(1.0 / dt))) if dt > 0 else 1
            new_length = (len(voltage) // ratio) * ratio
            if new_length > 0:
                soma_voltage_low = np.mean(voltage[:new_length].reshape(-1, ratio), axis=1)
            else:
                soma_voltage_low = voltage.copy()

            # convert raster back to dict of spike times per section
            # Training code expects: {synapse_idx: array_of_spike_times_in_ms}
            def raster_to_dict(raster, step_dt):
                spikes_dict = {}
                for idx in range(raster.shape[0]):
                    spike_indices = np.where(raster[idx] > 0)[0]
                    if len(spike_indices) > 0:
                        # Convert indices to time in ms (as integers)
                        spike_times = (spike_indices * step_dt).astype(int)
                        spikes_dict[idx] = spike_times.tolist()  # Convert to list for pickle compatibility
                    else:
                        spikes_dict[idx] = []  # Empty list if no spikes
                return spikes_dict

            ex_input_spikes = raster_to_dict(ex_input_raster, dt)
            inh_input_spikes = raster_to_dict(inh_input_raster, dt)
            
            # Ensure outputSpikeTimes is numpy array of floats (training code does: (arr.astype(float) - 0.5).astype(int))
            if isinstance(output_spikes, list):
                output_spikes = np.array(output_spikes, dtype=np.float32)
            else:
                output_spikes = output_spikes.astype(np.float32)

            converted_dict = {
                "somaVoltageHighRes": voltage.astype(np.float32) if isinstance(voltage, np.ndarray) else np.array(voltage, dtype=np.float32),
                "recordingTimeLowRes": np.arange(len(soma_voltage_low), dtype=int),
                "somaVoltageLowRes": soma_voltage_low.astype(np.float32),
                "exInputSpikeTimes": ex_input_spikes,
                "inhInputSpikeTimes": inh_input_spikes,
                "outputSpikeTimes": output_spikes,
            }

            return converted_dict

        except Exception as e:
            print(f"Error loading trial {trial_id}: {e}")
            return None

    def convert_from_trial_ids(self, trial_ids):
        all_sim_dicts = []
        trial_mapping = {}
        
        # Track simulation duration from first valid trial
        sim_duration_ms = None
        sim_duration_sec = None

        iterator = trial_ids
        if HAS_TQDM:
            iterator = tqdm.tqdm(trial_ids, desc="Processing reduce simulations")

        for sim_index, trial_id in enumerate(iterator):
            try:
                sim_dict = self.load_trial_data(trial_id)
                if sim_dict is None:
                    continue
                
                # Extract simulation duration from first valid trial
                if sim_duration_ms is None:
                    # Calculate duration from voltage trace length
                    voltage_low = sim_dict.get("somaVoltageLowRes", sim_dict.get("voltage", []))
                    if len(voltage_low) > 0:
                        # Assume 1 ms resolution for low-res voltage (as per pipeline)
                        sim_duration_ms = len(voltage_low)
                        sim_duration_sec = sim_duration_ms / 1000.0
                
                # Ensure outputSpikeTimes is numpy array (not list)
                if isinstance(sim_dict["outputSpikeTimes"], list):
                    sim_dict["outputSpikeTimes"] = np.array(sim_dict["outputSpikeTimes"])
                
                trial_mapping[sim_index] = {"trial_index": trial_id, "sim_index": sim_index}
                all_sim_dicts.append(sim_dict)
            except Exception as e:
                print(f"Error processing trial {trial_id}: {e}")

        # If duration is still None, try to infer from first simulation
        if sim_duration_sec is None and len(all_sim_dicts) > 0:
            first_sim = all_sim_dicts[0]
            voltage_low = first_sim.get("somaVoltageLowRes", [])
            if len(voltage_low) > 0:
                sim_duration_ms = len(voltage_low)
                sim_duration_sec = sim_duration_ms / 1000.0
        
        # Ensure duration is always set (required by training code)
        if sim_duration_sec is None:
            raise ValueError("Could not determine simulation duration from any trial. Please check data format.")
        
        # Set Params to match training code expectations
        # Training code expects: experiment_dict['Params']['totalSimDurationInSec'] * 1000
        sim_params = {
            "totalSimDurationInSec": float(sim_duration_sec),  # Must be a number, not None
            "totalSimDurationInMs": int(sim_duration_ms),
        }

        final_data = {
            "Params": sim_params,
            "Results": {"listOfSingleSimulationDicts": all_sim_dicts},
            "TrialMapping": trial_mapping,
        }
        return final_data

    def save_to_pickle(self, final_data, filename):
        with open(filename, "wb") as f:
            pickle.dump(final_data, f)

    def split_pickle_file(self, input_file="output.pkl", num_files=10):
        with open(input_file, "rb") as f:
            data = pickle.load(f)

        all_trials = data["Results"]["listOfSingleSimulationDicts"]
        trial_mapping = data.get("TrialMapping", {})
        total_trials = len(all_trials)
        if total_trials == 0:
            print("No trials to split.")
            return

        trials_per_file = total_trials // num_files
        for i in range(num_files):
            start = i * trials_per_file
            end = (i + 1) * trials_per_file if i < (num_files - 1) else total_trials

            sub_data = {
                "Params": data["Params"],
                "Results": {"listOfSingleSimulationDicts": all_trials[start:end]},
                "TrialMapping": {k: v for k, v in trial_mapping.items() if start <= k < end},
            }

            filename = self.output_dir / f"reduce_model_Output_spikes_{i:04d}.p"
            with open(filename, "wb") as f_out:
                pickle.dump(sub_data, f_out)

        print(f"Splitting completed into {num_files} files.")

    def organize_dataset(self, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
        files = sorted(self.output_dir.glob("*.p"))
        if not files:
            print(f"No .p files found in {self.output_dir}")
            return []

        total_files = len(files)
        train_count = int(total_files * train_ratio)
        valid_count = int(total_files * valid_ratio)

        train_files = files[:train_count]
        valid_files = files[train_count:train_count + valid_count]
        test_files = files[train_count + valid_count:]

        self._copy_files(train_files, self.train_dir)
        self._copy_files(valid_files, self.valid_dir)
        self._copy_files(test_files, self.test_dir)

        test_trial_indices = self._extract_trial_indices_from_files(test_files)

        print(f"Dataset organized:")
        print(f"  Training set: {len(train_files)} files")
        print(f"  Validation set: {len(valid_files)} files")
        print(f"  Test set: {len(test_files)} files")
        print(f"  Test set trial indices: {sorted(test_trial_indices)}")

        self._save_test_trial_indices(test_trial_indices)
        return test_trial_indices

    def _copy_files(self, file_list, target_dir):
        if HAS_TQDM:
            file_list = tqdm.tqdm(file_list, desc=f"Copying to {target_dir.name}")
        for file_path in file_list:
            shutil.copy2(file_path, target_dir / file_path.name)

    def _extract_trial_indices_from_files(self, file_list):
        all_trial_indices = []
        for file_path in file_list:
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                trial_mapping = data.get("TrialMapping", {})
                file_trial_indices = [mapping["trial_index"] for mapping in trial_mapping.values()]
                all_trial_indices.extend(file_trial_indices)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        return all_trial_indices

    def _save_test_trial_indices(self, test_trial_indices):
        test_indices_file = self.output_dir / "test_trial_indices.json"
        with open(test_indices_file, "w") as f:
            json.dump(
                {
                    "test_trial_indices": sorted(test_trial_indices),
                    "total_test_trials": len(test_trial_indices),
                },
                f,
                indent=2,
            )
        print(f"Test trial indices saved to: {test_indices_file}")

    def load_test_trial_indices(self):
        test_indices_file = self.output_dir / "test_trial_indices.json"
        if test_indices_file.exists():
            with open(test_indices_file, "r") as f:
                data = json.load(f)
            return data["test_trial_indices"]
        print(f"Test trial indices file not found: {test_indices_file}")
        return []

    def run_full_pipeline(self, trial_ids, num_files=10,
                          train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
        start_time = time.time()
        print(f"Starting NEURON-reduce dataset pipeline...")
        print(f"Total simulations: {len(trial_ids)}")

        # Step 1: convert all trials to one pickle
        print("Step 1: Converting simulation data...")
        final_data = self.convert_from_trial_ids(trial_ids)
        self.save_to_pickle(final_data, "output.pkl")
        print("Step 1 completed: output.pkl created")

        # Step 2: split into smaller files
        print("Step 2: Splitting pickle file...")
        self.split_pickle_file("output.pkl", num_files)

        # Step 3: organize train/valid/test
        print("Step 3: Organizing dataset...")
        test_trial_indices = self.organize_dataset(train_ratio, valid_ratio, test_ratio)

        total_time = time.time() - start_time
        print("Reduce pipeline completed successfully!")
        print(f"Total time: {total_time:.2f}s")
        return test_trial_indices


if __name__ == "__main__":
    # Example usage; adjust paths as needed
    root_folder_path = "neuron_reduce_simulations"  # where trial_*.pkl are saved
    # Align paths with IF pipeline outputs (adjust if needed)
    output_dir = "/G/results/aim2_sjc/Data/reduce_model_output_dataset"
    train_dir = "/G/results/aim2_sjc/Models_TCN/reduce_model_InOut/data/reduce_model_train"
    valid_dir = "/G/results/aim2_sjc/Models_TCN/reduce_model_InOut/data/reduce_model_valid"
    test_dir = "/G/results/aim2_sjc/Models_TCN/reduce_model_InOut/data/reduce_model_test"

    # Define trial IDs you want to convert
    num_trials = 10000
    trial_ids = list(range(1, 1+num_trials))  # example: first 100 trials

    pipeline = ReduceDatasetPipeline(
        root_folder_path=root_folder_path,
        output_dir=output_dir,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
    )

    pipeline.run_full_pipeline(
        trial_ids=trial_ids,
        num_files=max(1, len(trial_ids) // 1000),  # split roughly every 1000 trials
    )

