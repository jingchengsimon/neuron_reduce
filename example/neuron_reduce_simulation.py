"""Example script: reduce a detailed L5PC cell with Neuron_Reduce, run trials, and (optionally) plot."""

from __future__ import division

import argparse
import os
import time
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from neuron import h
import neuron_reduce


def build_reduced_cell():
    """Load L5PC model, add random synapses, and return reduced cell + input objects.

    Note: we must keep Python references to NetStim objects, otherwise their HOC
    instances may be garbage-collected and NetCon will lose its source.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_dir = os.path.join(script_dir, "modelFile")
    
    print(model_file_dir)

    # load hoc templates
    h.load_file(os.path.join(model_file_dir, "L5PCbiophys3.hoc"))
    h.load_file("import3d.hoc")
    h.load_file(os.path.join(model_file_dir, "L5PCtemplate.hoc"))

    complex_cell = h.L5PCtemplate(os.path.join(model_file_dir, "cell1.asc"))
    h.celsius = 37
    h.v_init = complex_cell.soma[0].e_pas

    synapses_list, netstims_list, netcons_list, randoms_list = [], [], [], []

    # choose segments (apical + basal) as synapse targets
    all_segments = [i for j in map(list, list(complex_cell.apical)) for i in j] + [
        i for j in map(list, list(complex_cell.basal)) for i in j
    ]
    len_per_segment = np.array([seg.sec.L / seg.sec.nseg for seg in all_segments])

    rnd = np.random.RandomState(10)
    n_syn = 10000
    for i in range(n_syn):
        seg_for_synapse = rnd.choice(all_segments, p=len_per_segment / sum(len_per_segment))
        syn = h.Exp2Syn(seg_for_synapse)

        # 85% exc, 15% inh
        if rnd.uniform() < 0.85:
            e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8, 1000 / 2.5, 0.0016
        else:
            e_syn, tau1, tau2, spike_interval, syn_weight = -86, 1, 8, 1000 / 15.0, 0.0008

        syn.e, syn.tau1, syn.tau2 = e_syn, tau1, tau2
        synapses_list.append(syn)

        ns = h.NetStim()
        ns.interval, ns.number, ns.start, ns.noise = spike_interval, 9e9, 100, 1
        netstims_list.append(ns)

        r = h.Random()
        r.Random123(i)
        r.negexp(1)
        ns.noiseFromRandom(r)
        randoms_list.append(r)

        nc = h.NetCon(ns, syn)
        nc.delay, nc.weight[0] = 0, syn_weight
        netcons_list.append(nc)

    # apply Neuron_Reduce
    reduced_cell, synapses_list, netcons_list = neuron_reduce.subtree_reductor(
        complex_cell, synapses_list, netcons_list, reduction_frequency=0
    )

    return reduced_cell, synapses_list, netcons_list, randoms_list, netstims_list


def run_trials(reduced_cell, synapses_list, netcons_list, randoms_list, netstims_list,
               num_trials=2, t_cut=200.0, t_window=6000.0, do_plot=True, do_save=True):
    """Run multiple trials; optionally plot per-trial input/output."""
    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}")

        # only change NetStim randomness between trials
        for i, r in enumerate(randoms_list):
            r.seq(trial + 2)

        # record spikes separately for exc / inh
        exc_section_spike_record = {}
        inh_section_spike_record = {}

        for nc, syn in zip(netcons_list, synapses_list):
            seg = syn.get_segment()
            sec = seg.sec
            sec_name = h.secname(sec=sec)

            is_exc = float(syn.e) == 0.0
            target_dict = exc_section_spike_record if is_exc else inh_section_spike_record

            if sec_name not in target_dict:
                target_dict[sec_name] = []

            vec = h.Vector()
            nc.record(vec)
            target_dict[sec_name].append(vec)

        # record soma output
        soma_v = h.Vector().record(reduced_cell.soma[0](0.5)._ref_v)
        t_vec = h.Vector().record(h._ref_t)

        # run simulation
        h.tstop = t_cut + t_window
        h.v_init = reduced_cell.soma[0].e_pas
        st = time.time()
        h.run()
        print("  reduced cell simulation time (trial {}) {:.4f}".format(trial + 1, time.time() - st))

        # convert spike records -> numpy, keep only times >= t_cut
        # 优化：使用 Vector 的 .to_python() 方法（如果可用），否则回退到 list()
        def vec_to_array(v):
            """Convert NEURON Vector to numpy array efficiently."""
            if hasattr(v, 'to_python'):
                return np.array(v.to_python())
            else:
                return np.array(list(v))
        
        exc_sec_spike_times = {}
        for sec_name, vec_list in exc_section_spike_record.items():
            if len(vec_list) == 0:
                exc_sec_spike_times[sec_name] = np.array([])
                continue
            # 批量转换：先收集所有非空 Vector，再一次性合并转 numpy
            spike_arrays = []
            for v in vec_list:
                if len(v) > 0:
                    spike_arrays.append(vec_to_array(v))
            spikes = np.concatenate(spike_arrays) if spike_arrays else np.array([])
            exc_sec_spike_times[sec_name] = spikes[spikes >= t_cut] if spikes.size > 0 else np.array([])

        inh_sec_spike_times = {}
        for sec_name, vec_list in inh_section_spike_record.items():
            if len(vec_list) == 0:
                inh_sec_spike_times[sec_name] = np.array([])
                continue
            spike_arrays = []
            for v in vec_list:
                if len(v) > 0:
                    spike_arrays.append(vec_to_array(v))
            spikes = np.concatenate(spike_arrays) if spike_arrays else np.array([])
            inh_sec_spike_times[sec_name] = spikes[spikes >= t_cut] if spikes.size > 0 else np.array([])

        # voltage trace, cut first t_cut ms
        # 优化：使用 to_python() 而不是 list()
        time_all = vec_to_array(t_vec)
        soma_all = vec_to_array(soma_v)
        valid_mask = time_all >= t_cut
        time_v = time_all[valid_mask]
        soma_voltage = soma_all[valid_mask]

        soma_spike_mask = soma_voltage > 0
        soma_spike_times = time_v[soma_spike_mask]

        # 统一 section 顺序（只计算一次，后续复用）
        section_names = sorted(
            set(list(exc_sec_spike_times.keys()) + list(inh_sec_spike_times.keys()))
        )
        num_sections = len(section_names)

        # ===== 保存当前 trial 为 pkl：构造 raster 并写入磁盘（可选） =====
        if do_save:
            # dt is fixed at 1/40000 seconds (0.025 ms) for NEURON simulations
            # This matches the dt used in 2_dataset_pipeline.py
            dt_sec = 1.0 / 40000.0  # seconds
            dt_ms = dt_sec * 1000.0  # milliseconds (0.025 ms)
            
            t_start = float(time_v[0]) if len(time_v) > 0 else t_cut
            duration = float(time_v[-1] - t_start) if len(time_v) > 0 else t_window

            # 优化：批量构造 spike-time 列表（相对窗口起点）
            exc_spike_list = []
            inh_spike_list = []
            for sec_name in section_names:
                exc_spikes = exc_sec_spike_times.get(sec_name, np.array([]))
                inh_spikes = inh_sec_spike_times.get(sec_name, np.array([]))
                # 向量化操作：一次性减去起点
                exc_spike_list.append((exc_spikes - t_start) if exc_spikes.size > 0 else np.array([]))
                inh_spike_list.append((inh_spikes - t_start) if inh_spikes.size > 0 else np.array([]))

            # 优化：spike-times -> raster (section × time)，使用向量化操作
            def spike_times_to_raster(spike_times_list, duration_ms, dt_ms, n_units):
                n_time_steps = int(np.ceil(duration_ms / dt_ms)) + 1  # 确保足够大
                raster = np.zeros((n_units, n_time_steps), dtype=np.float32)
                for unit_idx, spike_times in enumerate(spike_times_list):
                    if len(spike_times) == 0:
                        continue
                    # 向量化：一次性计算所有索引
                    spike_indices = np.round(spike_times / dt_ms).astype(int)
                    valid_mask = (spike_indices >= 0) & (spike_indices < n_time_steps)
                    if np.any(valid_mask):
                        raster[unit_idx, spike_indices[valid_mask]] = 1.0
                return raster

            ex_input_raster = spike_times_to_raster(
                exc_spike_list, duration, dt_ms, num_sections
            )
            inh_input_raster = spike_times_to_raster(
                inh_spike_list, duration, dt_ms, num_sections
            )

            # 输出 spike times（相对窗口起点）
            output_spike_times = (soma_spike_times - t_start).astype(np.float32)

            sim_dict = {
                "voltage": soma_voltage.astype(np.float32),
                "exInputSpikeTimes": ex_input_raster,
                "inhInputSpikeTimes": inh_input_raster,
                "outputSpikeTimes": output_spike_times,
            }

            save_dir = Path("neuron_reduce_simulations")
            save_dir.mkdir(exist_ok=True)
            pickle_path = save_dir / f"trial_{trial + 1}.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(sim_dict, f)
            print(f"  Saved trial {trial+1} data to {pickle_path}")

        if do_plot:
            plot_trials(trial, exc_sec_spike_times, inh_sec_spike_times,
                       soma_spike_times, time_v, soma_voltage)


def plot_trials(trial_index, exc_sec_spike_times, inh_sec_spike_times,
               soma_spike_times, time_v, soma_voltage):
    """Plot one trial: input spikes (exc/inh) and soma voltage."""
    from matplotlib.lines import Line2D

    # spike raster
    plt.figure()
    section_names = sorted(
        set(list(exc_sec_spike_times.keys()) + list(inh_sec_spike_times.keys()))
    )
    num_sections = len(section_names)

    for idx, sec_name in enumerate(section_names):
        # exc rows: 0..num_sections-1
        exc_row = idx
        exc_spikes = exc_sec_spike_times.get(sec_name, np.array([]))
        for t_spk in exc_spikes:
            plt.vlines(t_spk, exc_row - 0.4, exc_row + 0.4, colors="tab:blue", linewidth=1)

        # inh rows: num_sections+1..2*num_sections
        inh_row = idx + num_sections + 1
        inh_spikes = inh_sec_spike_times.get(sec_name, np.array([]))
        for t_spk in inh_spikes:
            plt.vlines(t_spk, inh_row - 0.4, inh_row + 0.4, colors="tab:green", linewidth=1)

    # soma spikes at y=-1
    for t_spk in soma_spike_times:
        plt.vlines(t_spk, -1 - 0.4, -1 + 0.4, colors="red", linewidth=1.0, alpha=0.7)

    exc_rows = list(range(num_sections))
    inh_rows = list(range(num_sections + 1, 2 * num_sections + 1))
    yticks = [-1] + exc_rows + inh_rows
    yticklabels = (
        ["soma"]
        + [f"exc_{i}" for i in range(num_sections)]
        + [f"inh_{i}" for i in range(num_sections)]
    )
    plt.yticks(yticks, yticklabels)

    legend_elements = [
        Line2D([0], [0], color="tab:blue", lw=1, label="exc spikes"),
        Line2D([0], [0], color="tab:green", lw=1, label="inh spikes"),
        Line2D([0], [0], color="red", lw=1, label="soma spikes"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.xlabel("time (ms)")
    plt.ylabel("section index")
    plt.title(f"Spike times per reduced section and soma (trial {trial_index + 1})")
    plt.tight_layout()

    # soma voltage
    plt.figure()
    plt.plot(time_v, soma_voltage, label=f"trial {trial_index + 1}")
    plt.xlabel("time (ms)")
    plt.ylabel("voltage (mV)")
    plt.title(f"Soma voltage (trial {trial_index + 1})")
    plt.legend()
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(
        description="Run Neuron_Reduce example with optional plotting."
    )
    parser.add_argument("--trials", type=int, default=1, help="number of trials")
    parser.add_argument("--t_cut", type=float, default=200.0, help="ignore initial time (ms)")
    parser.add_argument("--window", type=float, default=6000.0, help="simulation time (ms)")
    parser.add_argument("--plot", type=str, default="true", help="whether to plot results")
    parser.add_argument("--save", type=str, default="true", help="whether to save pkl files")
    args = parser.parse_args()

    plot_flag = False # str(args.plot).lower() in ("true", "1", "yes", "y")
    save_flag = True # str(args.save).lower() in ("true", "1", "yes", "y")

    reduced_cell, synapses_list, netcons_list, randoms_list, netstims_list = build_reduced_cell()
    run_trials(
        reduced_cell,
        synapses_list,
        netcons_list,
        randoms_list,
        netstims_list,
        num_trials=args.trials,
        t_cut=args.t_cut,
        t_window=args.window,
        do_plot=plot_flag,
        do_save=save_flag,
    )

    if plot_flag:
        plt.show()


if __name__ == "__main__":
    main()
