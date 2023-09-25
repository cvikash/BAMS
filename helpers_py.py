import numpy as np
import h5py
import glob
import socket
import pickle 
import pymongo as pym
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt

MONGO_URL = "mongodb://10.48.10.5:30005,10.48.10.6:30006,10.48.10.8:30008,10.48.10.11:30011,10.48.10.14:30014/?replicaSet=devrs"

# ================================ BEHAVIOR HELPERS ============================
def get_displacement(x_fish, y_fish, fs, window_len, pix_size_um):
    speed = get_speed(x_fish, y_fish)

    window_bins = int(fs * window_len)
    displacement = np.nan * np.ones(x_fish.shape)
    for i in range(x_fish.shape[0]):
        if (i + window_bins) > x_fish.shape[0]:
            displacement[i] = displacement[i-1]
        else:
            _d = sum(speed[i:(i + window_bins)])
            displacement[i] = _d
    displacement = displacement * (pix_size_um*1e-3)
    return displacement

def get_speed(x, y):
    return np.append(np.sqrt((np.diff(x)**2)+(np.diff(y)**2)), 0)


# ================================ PYHSMM HELPERS ===============================
def load_pyhsmm(filename):
    """
    load pickled model fit
    """
    with open(filename, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def smoothed_viterbi(margin_prob, lags=1):
    """
    margin_prob: T x N-states
    lags: Integer number of history to use
    output: state sequence after smoothin (length T vector)
    """
    smoothed_margin_prob = np.zeros(margin_prob.shape)
    weight = np.repeat(1.0, lags) / lags
    
    for i in range(margin_prob.shape[1]):
        smoothed_margin_prob[:,i] = np.convolve(margin_prob[:,i], weight, 'same')
    
    return np.argmax(smoothed_margin_prob, axis = 1), smoothed_margin_prob    # returns stateseq after smoothing the state marginals

def slice_periods(stat_seq, slice_stat):
    """
    stat_seq: 1 x T
    slice_state: 0 or 1
    output: start and end index of slice_state
    """
    try:
        if slice_stat==0:
            down_idx = np.argwhere(np.diff(stat_seq)<0)[:,0] 
            up_idx = np.argwhere(np.diff(stat_seq)>0)[:,0]
            if (len(up_idx)==0) & (len(down_idx)==0) & (stat_seq[-1]!=0):
                return [], []
            elif len(up_idx) == 0:
                up_idx = np.array([len(stat_seq)])
            elif len(down_idx) == 0:
                down_idx = np.array([0])
        else:
            down_idx = np.argwhere(np.diff(stat_seq)>0)[:,0]
            up_idx = np.argwhere(np.diff(stat_seq)<0)[:,0]
            
            if (len(up_idx)==0) & (len(down_idx)==0) & (stat_seq[-1]!=1):
                return [], []
            elif len(down_idx) == 0:
                down_idx = np.array([0])
            elif len(up_idx) == 0:
                up_idx = np.array([len(stat_seq)]) 

        if down_idx[0] < up_idx[0]:
            if len(down_idx) != len(up_idx):
                start_idx = down_idx
                end_idx = np.append(up_idx,len(stat_seq)) +1
            else:
                start_idx = down_idx
                end_idx = up_idx +1
        else:
            down_idx = np.append(0,down_idx)
            if len(down_idx) != len(up_idx):
                start_idx = down_idx
                end_idx = np.append(up_idx,len(stat_seq)) +1
            else:
                start_idx = down_idx
                end_idx = up_idx +1
        return start_idx, end_idx
    except:
        return [],[]
        

def get_duration_histogram(state_sequences, nstates, bins, fs, bootstrap=False, nboots=100, bootsamples=500):
    """
    Return histogram of state durations for the given state sequence.

        - state_sequences is a list of state_sequence np.arrays, one entry per fish
        - nstates: int, number of states
        - bins specifies how to bin the durations (in minutes)
        - fs = sampling rate of the datasets

    Return:
        bins = np.array of bins for the histogram
        histogram = dictionary, keys=states, values=histogram for each state
    """
    # allocate dict to hold durations of each unique state
    labels = np.arange(0, nstates)
    durations = {l: [] for l in labels}
    boot_error = {l: [] for l in labels}
    histograms = dict.fromkeys(labels)
    for lab in labels:
        for state_seq in state_sequences:
            bb = (state_seq == lab).astype(np.int)
            ss, ee = get_state_boundaries(bb)            
            try:
                for (s, e) in zip(ss, ee):
                    durations[lab] = np.append(durations[lab], [(e / fs / 60) - (s / fs / 60)])
            except TypeError:
                durations[lab] = np.append(durations[lab], [(ee / fs / 60) - (ss / fs / 60)])
        # make the duration histogram
        if bootstrap:
            # compute "nboots" resamples of the durations to generate the histogram
            # return the average across bootstraps, and return the bootstrap standard error (sd across boots)
            vals = np.zeros((nboots, len(bins)-1))
            for nboot in range(nboots):
                dur = np.random.choice(np.arange(0, len(durations[lab])), bootsamples, replace=True)
                v, _ = np.histogram(durations[lab][dur], bins=bins)
                vals[nboot, :] = v / bootsamples  

            histograms[lab] = np.mean(vals, axis=0)
            boot_error[lab] = np.std(vals, axis=0)
        else:
            vals, _ = np.histogram(durations[lab], bins=bins)
            vals = vals / len(durations[lab])
            histograms[lab] = vals
    if bootstrap:
        return bins, histograms, boot_error
    else:
        return bins, histograms


def get_emission_probabilities(state_sequences, nstates):
    """
    Return probability of observing each state

        state_sequences = list of state sequences. One per fish.
        nstates = int, number of states in the model
    
    Return:
        probabilities = dict, keys=state, values=probability
    """
    # concatenate state_sequences into one long vector
    seq = np.concatenate(state_sequences)
    labels = np.arange(0, nstates)
    probabilities = dict.fromkeys(labels)
    for lab in labels:
        count = np.sum(seq==lab)
        prob = count / len(seq)
        probabilities[lab] = prob
    return probabilities

def remove_short_states(seq, min_dur):
    """
    Replace short states (duration < min_dur) with NaN
    """
    u = np.unique(seq)
    for uu in u:
        bb = seq == uu
        ss, ee = get_state_boundaries(bb)
        if ss.size>=1:
            try:
                for s, e in zip(ss, ee):
                    if (e - s) < min_dur:
                        seq[s:(e+1)] = np.nan
            except:
                for s, e in zip([ss], [ee]):
                    if (e - s) < min_dur:
                        seq[s:(e+1)] = np.nan
    return seq
# ================================================================================

# ================================ GENERAL HELPERS ===============================

def unpackh5(filename, variables=False):
    f = h5py.File(filename, "r")
    r = dict()
    if variables==False:
        for k in f.keys():
            try:
                r[k] = np.array(f.get(k), dtype="float64")
            except:
                r[k] = np.array(f.get(k))
            
    else:
        for v in variables:
            try:
                r[v] = np.array(f.get(v), dtype="float64")
            except:
                r[v] = np.array(f.get(v))
    return r

   
def locate_all(timestamp, server, filename, raw=False):
    root_dir = "/data/"
    if server != socket.gethostname():
        sidx = server.split("-")[-1]
        root_dir = f"/nfs/data{sidx}/"
    else:
        root_dir = "/data/"
    if raw:
        datadir = "data_raw"
    else:
        datadir = "data"
    
    return glob.glob(f"{root_dir}*/{datadir}/{timestamp}/{filename}")    


def locate(timestamp, server, filename, raw=False):
    files = locate_all(timestamp, server, filename, raw=raw)
    if len(files) > 1:
        raise ValueError("More than one match for $filename found on server $server: \n $files")
    
    return files[0]


# db query.
def dbquery(fields, filt):
    """
    Return dataframe with fields for each dataset matching filter.
    filter is dictionary of query filters
    fields is list of strings
    """
    client = pym.MongoClient(MONGO_URL)
    db = client.rolidb
    collection = db.data
    documents = collection.find(filt)
    df = []
    for doc in documents:
        data = []
        for k in fields:
            data.append(doc[k])
        df.append(data)
    df = pd.DataFrame(columns=fields, data=np.array(df))
    return df


# ================== PLOTTING HELPERS ========================
def compute_ellipse(x, y):
    inds = np.isfinite(x) & np.isfinite(y)
    x = np.array(x)[inds]
    y = np.array(y)[inds]
    data = np.vstack((x, y))
    mu = np.mean(data, 1)
    data = data.T - mu

    D, V = np.linalg.eig(np.divide(np.matmul(data.T, data), data.shape[0] - 1))
    # order = np.argsort(D)[::-1]
    # D = D[order]
    # V = abs(V[:, order])

    t = np.linspace(0, 2 * np.pi, 100)
    e = np.vstack((np.sin(t), np.cos(t)))  # unit circle
    VV = np.multiply(V, np.sqrt(D))  # scale eigenvectors
    e = np.matmul(VV, e).T + mu  # project circle back to orig space
    e = e.T

    return e


def nan_interpolate(X):
    x_new = X.copy()
    full_t = np.arange(0, X.shape[1])
    for ii in range(X.shape[0]):
        this_t = np.argwhere(np.isnan(X[ii, :])==False).squeeze()
        f = scipy.interpolate.interp1d(this_t, X[ii, this_t], fill_value="extrapolate")
        x_new[ii, :] = f(full_t)
    return x_new


def get_state_boundaries(seq_bool):
    ss = np.argwhere(np.diff(seq_bool.astype(int))>0).squeeze()
    ee = np.argwhere(np.diff(seq_bool.astype(int))<0).squeeze()
    
    # deal with all the weird different potential numpy types
    if (ee.size>0) & (ss.size>0):

        if (ss.shape==()) & (ee.shape==()):
            if ss > ee:
                ss = np.append([0], ss)
        elif (ss.shape==()) & (ee.shape!=()):
            if ss > ee[0]:
                ss = np.append([0], ss)
        elif (ee.shape==()) & (ss.shape!=()):
            if ss[0] > ee:
                ss = np.append([0], ss)
        else:
            if ss[0] > ee[0]:
                ss = np.append([0], ss)
            
    if ee.size < ss.size:
        ee = np.append(ee, [len(seq_bool)])
    
    elif ss.size < ee.size:
        ss = np.append([0], ss)

    ss = ss.squeeze()
    ee = ee.squeeze()
        
    return ss, ee


# density map
def get_3d_kernel(width=11, cutoff=5):
    distance_all = np.zeros((width, width, width))
    x_index, y_index, z_index = np.meshgrid(np.arange(0, distance_all.shape[0]),
                    np.arange(0, distance_all.shape[1]), 
                    np.arange(0, distance_all.shape[2])
    )
    for i in range(x_index.size):
        x = x_index.flatten()[i]
        y = y_index.flatten()[i]
        z = z_index.flatten()[i]
        distance_all[x, y, z] = np.linalg.norm([x+1-np.ceil(distance_all.shape[0] / 2), 
                                    y+1-np.ceil(distance_all.shape[1] / 2),
                                    z+1-np.ceil(distance_all.shape[2] / 2)])

    kernel = distance_all <= cutoff
    return kernel

def get_count(count_mat, indices):
    vals = np.zeros(len(indices))
    for i in range(len(indices)):
        vals[i] = count_mat[indices[i][0], indices[i][1], indices[i][2]]
    return vals

def get_3d_spatial_map(rois, stack_shape, kernel_width=11, k_cutoff=5):
    count = np.zeros(stack_shape).astype(int);
    for i in range(rois.shape[1]):
        x = int(np.round(rois[0, i]))
        y = int(np.round(rois[1, i]))
        z = int(np.round(rois[2, i]))
        count[x,y,z] += 1

    # get kernel
    kernel = get_3d_kernel(width=kernel_width, cutoff=k_cutoff)

    # get "density" map by convolving
    print("convolving counts with kernel...")
    spatial_p_img_convolved = scipy.ndimage.convolve(count, kernel)

    idx = [[rois[0, i].astype(int), rois[1, i].astype(int), rois[2, i].astype(int)] \
                    for i in range(rois.shape[1])]
    return spatial_p_img_convolved, idx

def get_density_count_per_roi(rois, stack_shape, kernel_width=11, k_cutoff=5):
    """
    For each ROI, return the N neurons at that location
    in a z/y/z projection.
    Estimate by counting neurons in each bin and convolving that
    count matrix with our spherical kernel

    Input:
        rois: 3 x nROI matrix (x/y/z locations)
    Return:
        dictionary:
            x_projection counts
            y_projection counts
            z_projection counts
    """
    spatial_p_img_convolved, idx = get_3d_spatial_map(rois, stack_shape, kernel_width, k_cutoff)

    return get_count(spatial_p_img_convolved, idx)


# ================================ NMF HELPERS =================================
def get_Aall_corrected(ds, server, corr_file="posprocessed_nmf.h5"):
    corr = locate(ds, server, corr_file)
    corr = unpackh5(corr)

    nmf = locate(ds, server, "NMF.h5")
    nmf = unpackh5(nmf, variables=["A_all"])
    
    # select good rois
    X = nmf["A_all"][corr["final_roi_indices"].astype(int)-1, :]
    del nmf

    # only correct for the "good" fits. Otherwise just subtract mean
    t = np.arange(0, X.shape[1])+1
    tau = corr["exp_tau"]
    amp = corr["exp_amp"]
    bmask = (tau < 0) | (tau > 0.01)
    tau[bmask] = 0
    amp[bmask] = 0
    print("Baseline correcting...")
    for i in np.arange(0, X.shape[0]):
        bb = amp[i] * np.exp(t * -1 * tau[i]) + corr["exp_offset"][i]
        X[i, :] = X[i, :] - bb
    
    return X, corr["final_roi_indices"].astype(int)


def downsample(x, fs_in, fs_out):
    step = int(fs_in / fs_out)
    t = np.arange(0, len(x), step)
    x_resampled = np.array([np.mean(x[ts:ts+step]) for ts in t])
    np.append(x_resampled, np.mean(x[t[-1]:]))    
    return x_resampled

def find_blocks(sequence, element):
    start_indices = []
    end_indices = []
    in_block = False

    for i, e in enumerate(sequence):
        if e == element:
            if not in_block:
                start_indices.append(i)
                in_block = True
        else:
            if in_block:
                end_indices.append(i - 1)
                in_block = False

    if in_block:
        end_indices.append(len(sequence) - 1)

    return list(zip(start_indices, end_indices))

def compute_transition_probabilities(sequence, block_bool):
    el_counter = np.zeros(3)
    el_counter[0] = np.sum(sequence==1)/len(sequence)
    el_counter[1] = np.sum(sequence==2)/len(sequence)
    el_counter[2] = np.sum(sequence==3)/len(sequence)
    if block_bool:
        blocks = []
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                blocks.append(sequence[i-1])
        blocks += [sequence[-1]]
    else:
        blocks = sequence
        
    elements = np.unique(blocks)
    elements  = elements[~np.isnan(elements)]
    trans_mat = np.nan*np.zeros((len(elements), len(elements)))
    if len(elements) <=1:
        return np.nan*np.zeros((3,3)),elements, el_counter*100
    for i, v in enumerate(elements):
        bin_loc = np.where(blocks == v)[0] + 1
        if bin_loc[-1] > len(blocks)-1:
            bin_loc = bin_loc[:-1]
            # count-=1
        next_elements = np.array(blocks)[bin_loc]
        count = np.sum(np.isfinite(next_elements))
        next_elem_counts = np.array([np.sum(next_elements == e) for e in elements])
        if count==0:
            trans_mat[i] = np.nan
        else:
          trans_mat[i] = next_elem_counts / count
    return trans_mat, elements, el_counter*100

def remove_short_periods(sequence, min_len):
    prev_el = 0
    ss = np.copy(sequence)
    for i in range(1,len(ss)):
        if ss[prev_el] ==ss[i]:
            continue
        else:
            if i-prev_el > min_len:
               prev_el = i
            else:
               ss[prev_el:i] = np.nan
               prev_el = i
    return ss


def hmm_marginal_probs(disp, saccade_rate, model_disp, model_saccade,nlags, slice_state):
    state_marginals_disp = model_disp.heldout_state_marginals(disp[:, np.newaxis])
    smoothed_state_seq, state_marginals_disp_smooth = smoothed_viterbi(state_marginals_disp, lags=nlags)
    start, end = slice_periods(smoothed_state_seq, slice_state)
    state_marginals_saccade = np.NaN*np.zeros(state_marginals_disp.shape)
    state_saccade_seq = np.ones(state_marginals_disp.shape[0])
    for j in range(len(start)):
        marginals = model_saccade.heldout_state_marginals(saccade_rate[start[j]:end[j], np.newaxis])

        if marginals.shape[0] <nlags:
            state_saccade_seq[start[j]:end[j]], state_marginals_saccade[start[j]:end[j],:] = smoothed_viterbi(marginals, lags=marginals.shape[0]/2)
        else:
            state_saccade_seq[start[j]:end[j]], state_marginals_saccade[start[j]:end[j],:] = smoothed_viterbi(marginals, lags=nlags)

    return state_marginals_disp_smooth, smoothed_state_seq, state_marginals_saccade, state_saccade_seq +2

def return_state_seq(disp, saccade_rate, model_disp, model_saccade,nlags):
    state_marginals_disp = model_disp.heldout_state_marginals(disp[:, np.newaxis])
    smoothed_state_seq, _ = smoothed_viterbi(state_marginals_disp, lags=nlags)
    start, end = slice_periods(smoothed_state_seq, 0)
    state_saccade_seq = np.ones(len(smoothed_state_seq))
    for j in range(len(start)):
        marginals = model_saccade.heldout_state_marginals(saccade_rate[start[j]:end[j], np.newaxis])

        if marginals.shape[0] <nlags:
            state_saccade_seq[start[j]:end[j]]= smoothed_viterbi(marginals, lags=marginals.shape[0]/2)[0] +2
        else:
            state_saccade_seq[start[j]:end[j]] = smoothed_viterbi(marginals, lags=nlags)[0] +2
    
    return  state_saccade_seq 
