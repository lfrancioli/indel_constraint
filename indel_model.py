import sys
import vcf
import argparse
import numpy as np
import logging
import random
from collections import OrderedDict
import math
import matplotlib
import statsmodels.formula.api as smf
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from pprint import pprint
from os import listdir, makedirs, path
import pickle
import itertools
import pandas as pd

from Bio import SeqIO

from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Input, Dense, Dropout, Flatten, Reshape, merge, SpatialDropout1D, concatenate
from keras import backend as K
from keras.layers.convolutional import Conv1D, MaxPooling1D
import tensorflow as tf

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("indel")
logger.setLevel(logging.INFO)

BASES = ['A','C','G','T']
dna_encoder = LabelBinarizer(sparse_output=True).fit(BASES)

def overlaps_intervals(intervals, start, stop):
    interval_start = np.searchsorted(intervals[:, 0], start)
    interval_stop = np.searchsorted(intervals[:, 0], stop)

    return (
        interval_start != interval_stop or
        start < intervals[interval_start - 1][1] or intervals[interval_start][0] == start or
        stop < intervals[interval_stop - 1][1] or intervals[interval_stop][0] == stop
    )

def get_interval_overlaps(intervals, start, stop):
    interval_start = max(0, np.searchsorted(intervals[:, 0], start) -1 )
    interval_stop = np.searchsorted(intervals[:, 0], stop)

    results = []
    for interval_start, interval_stop in intervals[interval_start:interval_stop]:
        if interval_start > start:
            results.append((start, interval_start, False))
        if interval_stop > start:
            results.append((max(start, interval_start), min(interval_stop, stop), True))
        start = interval_stop

    if start < stop:
        results.append((start, stop, False))

    return results


def in_intervals(intervals, value):
    interval_index = np.searchsorted(intervals[:, 0], value)
    return value < intervals[interval_index - 1][1] or intervals[interval_index][0] == value


def rle_encode_ambiguous(seq_list):
    ambiguous_indices = np.invert(np.in1d(seq_list, np.array(BASES, dtype='|S1')))
    pos, = np.where(np.diff(ambiguous_indices) != 0)
    pos = np.concatenate(([0], pos + 1, [len(ambiguous_indices)]))
    return np.array([(a, b) for (a, b) in zip(pos[:-1], pos[1:]) if ambiguous_indices[a]])


def load_fasta_reference(in_path):
    reference = {}
    ambiguous_bases = {}
    logger.info("Reading fasta file {}".format(in_path))
    start = timer()
    fasta = SeqIO.to_dict(SeqIO.parse(in_path, "fasta"))
    logger.info("Reading fasata file loaded in {}.".format(timer() - start))
    for contig, sequence in fasta.iteritems():
        start = timer()
        logger.info("Encoding chromosome {}".format(contig))
        seq_list = list(str(sequence.seq))
        one_hot = dna_encoder.transform(seq_list)
        reference[contig] = one_hot.astype(np.int8).todense()
        ambiguous_bases[contig] = rle_encode_ambiguous(seq_list)
        logger.info("Chrom {} encoded in {}.".format(contig, timer() - start))
    return reference, ambiguous_bases


def write_bitpacked_reference(reference, ambiguous_bases, out_path):
    logger.info("Writing reference to {}".format(out_path))
    if not path.exists(out_path):
        makedirs(out_path)
    for contig, sequence in reference.iteritems():
        np.save(out_path + "/{}.npy".format(contig), np.packbits(sequence))

    with open(out_path + "/ambiguous_bases.pickle", "w") as file:
        pickle.dump(ambiguous_bases, file)


def load_bitpacked_reference(in_path):
    start = timer()
    logger.info("Loading bitpacked reference from {}".format(in_path))
    reference = {}
    for f in listdir(in_path):
        if f.endswith(".npy"):
            packed = np.load(in_path + "/" + f)
            reference[f[:-4]] = np.unpackbits(packed).reshape(packed.shape[1]*2, 4)

    with open(in_path + "/ambiguous_bases.pickle", "r") as file:
        ambiguous_bases = pickle.load(file)

    for contig in reference:
        if contig not in ambiguous_bases:
            logger.warn("No ambiguous bases indices found for contig {}.".format(contig))

    logger.info("Reference loaded in {}.".format(timer() - start))
    return reference, ambiguous_bases

def load_coverage(in_path):
    start = timer()
    logger.info("Loading coverage files from {}".format(in_path))
    coverage = {}
    for f in listdir(in_path):
        if f.endswith(".npy"):
            coverage_values = np.load(in_path + "/" + f)
            coverage[f[:-4]] = coverage_values.reshape(len(coverage_values),1)

    logger.info("Coverage files loaded in {}".format(timer() - start))
    return coverage


def main(args):
    if args.ref_fasta:
        reference, ambiguous_bases = load_fasta_reference(args.ref_fasta)
        write_bitpacked_reference(reference, ambiguous_bases, args.ref_fasta + ".bp")
    else:
        reference, ambiguous_bases = load_bitpacked_reference(args.ref)

    if args.train_vcf is not None:
        training_indels, training_indel_contigs = read_vcfs(args.train_vcf, args.max_training, args.pass_only)

    if args.eval_vcf is not None:
        eval_indels, eval_indel_contigs = read_vcfs(args.eval_vcf, args.max_training, args.pass_only)
    elif args.train_vcf is not None:
        eval_indels, eval_indel_contigs = training_indels, training_indel_contigs

    if args.train_model or args.save_best_examples or args.compute_predictions:
        coverage = load_coverage(args.coverage)

    if args.train_model or args.save_best_examples:
        logger.info("Creating training tensors")
        training_bases, training_coverage, labels, indel_indices = vcf_to_indel_tensors(training_indels,
                                                               reference,
                                                               ambiguous_bases,
                                                               coverage,
                                                               args.window_size,
                                                               args.neg_train_window,
                                                               args.neg_to_pos_training)
        logger.info("Training tensors created with dimensions:\ntraining_bases: {}\ntraining coverage:{}\nlabels: {}\nindel indices: {}.".format(training_bases.shape, training_coverage.shape, labels.shape, indel_indices.shape))

        logger.info("Splitting training data into training and testing sets.")
        train, test = split_data([training_bases, training_coverage, labels, indel_indices], [0.8,0.2])

        logger.info("Training bases shape: {}, training coverage shape: {}, labels shape: {}".format(train[0].shape, train[1].shape, train[2].shape))

    K.set_learning_phase(1)
    model = make_indel_model(args.window_size)

    if args.train_model:
        K.set_learning_phase(1)
        logger.info("Training model. Learning phase =  {}".format(K.learning_phase()))
        if args.callback:
            model.load_weights(args.callback, by_name=True)
        history = model.fit([train[0], train[1]], train[2], batch_size=32, validation_split=0.2, shuffle=True, epochs=10, callbacks=get_callbacks(args.output+".callbacks"))

        logger.info("Plotting metrics.")
        plot_metric_history(history, args.output + ".metrics_history.pdf", "Indel model")

        K.set_learning_phase(0)
        logger.info("Evaluating model. Learning phase =  {}".format(K.learning_phase()))
        eval_values = model.evaluate([test[0], test[1]], test[2])
        pprint(", ".join(["{}: {}".format(metric, value) for metric, value in zip(model.metrics_names, eval_values)]))

    else:
        model.load_weights(args.callback, by_name=True)

        K.set_learning_phase(0)

    if args.save_best_examples:
        logger.info("Saving best scoring training indel examples. Learning phase =  {}".format(K.learning_phase()))
        positive_examples, negative_examples = get_best_scoring_training_examples(model, train)
        with open(args.output + ".best_training_examples.tsv",'w') as file:
            for seq, cov, score, label, indel_index in positive_examples + negative_examples:
                ref = training_indels[indel_index].REF if indel_index > -1 else seq[30]
                alt = ",".join([str(a) for a in training_indels[indel_index].ALT]) if indel_index > -1 else "null"
                training_label = "indel" if label[0] == 1 else "no_indel"
                file.write("{}\t{}\t{}\t{}\t{}\t{}\t{:.3f}\t{}\n".format(seq[:30], seq[30], seq[31:], ref, alt, cov, score, training_label))

    predictions = {}
    predictions_path = args.output + ".predictions/"
    if args.compute_predictions:
        if not path.exists(predictions_path):
            makedirs(predictions_path)

        if args.predictions_interval:
            pred_contig, pred_end = args.predictions_interval.split(":")
        else:
            pred_contig = None
            pred_end = None

        for contig in reference.keys():
            if pred_contig is None or pred_contig == contig:
                reference_contig = reference[contig] if pred_end is None else reference[contig][0:int(pred_end)]
                results = apply_model(model, reference_contig, ambiguous_bases[contig], coverage[contig], args.window_size)
                predictions[contig] = results
                np.save(predictions_path + "{}.npy".format(contig), results)

    if args.load_predictions:
        for f in listdir(predictions_path):
            if f.endswith(".npy"):
                predictions[f[:-4]] = np.load(predictions_path + f)
        logger.info("Loaded predictions from {}:\n{}".format(
            predictions_path,
            "\n".join(["{}: {}".format(c,len(b)) for c,b in predictions.iteritems()])))

    if args.eval_predictions:
        logger.info("Evaluating predictions.")
        results = eval_predictions(predictions, eval_indels, eval_indel_contigs, ambiguous_bases, args.eval_bin_size)
        if results is not None:
            plot_binned_eval(results, args.output + ".linreg_binned_{}.pdf".format(args.eval_bin_size))


def read_vcfs(vcf_paths, max_training, pass_only):
    indels = []
    contigs = OrderedDict()

    for vcf in vcf_paths:
        vcf_indels, vcf_contigs = read_vcf(vcf, max_training - len(indels), pass_only)
        next_indel_index = contigs[contigs.keys()[-1]][-1] + 1 if contigs else 0
        for contig in vcf_contigs.keys():
            if contig in contigs.keys():
                sys.exit("FATAL: Contig {} found in multiple VCF files (last file: {}). Each contig has to be contained within a single VCF file.". format(contig, vcf))
            contigs[contig] = [i + next_indel_index for i in vcf_contigs[contig]]

        indels.extend(vcf_indels)

        if max_training > 0 and len(indels) >= max_training:
            break

    return indels, contigs


def read_vcf(vcf_path, max_training, pass_only):
    logger.info("Loading indels from VCF {}.".format(vcf_path))
    indels = []
    contigs = OrderedDict()

    with open(vcf_path, 'r') as file:
        vcf_reader = vcf.Reader(file)

        # Get positive training examples
        start = timer()
        for variant in vcf_reader:
            if max_training > 0 and len(indels) >= max_training:
                break

            if not variant.is_snp and (not pass_only or not variant.FILTER):
                if not variant.CHROM in contigs:
                    if contigs:
                        contigs[contigs.keys()[-1]].append(len(indels) -1)
                    contigs[variant.CHROM] = [len(indels)]
                indels.append(variant)

        contigs[contigs.keys()[-1]].append(len(indels) - 1)

        logger.info("Loaded indels from VCF {} in {}:\n{}".format(vcf_path,
                                                                  timer() - start,
                                                                  "\n".join("{}: {}".format(contig, end-start+1) for contig, (start,end) in contigs.iteritems())))

    return indels, contigs

def vcf_to_indel_tensors(indels, reference, ambiguous_bases, coverage, window_size, neg_train_window, neg_to_pos_training, squash_multi_allelic = False):
    positive_training_bases = []
    negative_training_bases = []
    positive_training_coverage = []
    negative_training_coverage = []
    labels = np.empty((0,2),dtype=np.int8)
    indel_indices = []

    # Get positive training examples
    start = timer()
    positive_training_pos = {}
    contig = indels[0].CHROM
    for i, variant in enumerate(indels):
        #Get positive training examples
        if not variant.is_snp and not (squash_multi_allelic and variant.POS in positive_training_pos):
            pos = variant.POS
            positive_training_bases.append(reference[contig][pos - window_size -1: pos + window_size])
            positive_training_coverage.append(coverage[contig][pos - window_size -1: pos + window_size])
            indel_indices.append(i)
            while pos in positive_training_pos: # A bit of a trick for multi-allelic indels, but should be fine
                pos+=1
            positive_training_pos[pos] = 1

        if i+1 == len(indels) or indels[i+1].CHROM != variant.CHROM:
            print i
            # Get negative training examples
            neg_positions = get_negative_training_positions(positive_training_pos,
                                                                           ambiguous_bases[contig],
                                                                           len(reference[contig]) - window_size,
                                                                           neg_train_window,
                                                                           window_size,
                                                                           neg_to_pos_training
                                                                           )
            negative_training_bases.extend([reference[contig][neg_pos - window_size -1: neg_pos + window_size] for neg_pos in neg_positions])
            negative_training_coverage.extend(
                [coverage[contig][neg_pos - window_size - 1: neg_pos + window_size] for neg_pos in neg_positions])

            indel_indices.extend([-1]*len(positive_training_pos) * neg_to_pos_training)
            labels = np.append(
                labels,
                np.concatenate(
                (
                    np.full((len(positive_training_pos), 2), np.array([1, 0])),  # positive labels
                    np.full((len(positive_training_pos) * neg_to_pos_training, 2), np.array([0, 1]))  # negative labels
                ), axis=0),
                axis=0
            )

            positive_training_pos = {}
            contig = variant.CHROM

    logger.info("Training examples loaded in {}.".format(timer() - start))

    return np.asarray(positive_training_bases + negative_training_bases), np.asarray(positive_training_coverage + negative_training_coverage), labels, np.asarray(indel_indices)

def one_hot_encode(data, encoder):
    all_data = "".join(data)
    res = encoder.transform(list(all_data))
    return np.asarray(np.split(res, len(data)))


def get_training_tensor(chrom, pos, reference, window_size, encoder):
    contig = reference[chrom]
    record = contig[pos - window_size: pos + window_size + 1]
    return encoder.transform(list(record))


def get_negative_training_positions(positions, ambiguous_bases, max_pos, neg_train_window, window_size, neg_to_pos_training, max_retry = 100):
    neg_positions = []
    if neg_train_window > 0:
        for pos in positions.keys():
            for i in range(neg_to_pos_training):
                neg_pos = random.randint(pos - neg_train_window, pos + neg_train_window)
                i = 0
                while (neg_pos in positions or in_intervals(ambiguous_bases, neg_pos) or neg_pos > max_pos or neg_pos < window_size) and i < max_retry:
                    neg_pos = random.randint(pos - neg_train_window, pos + neg_train_window)
                    i+=1

                if i == max_retry:
                    logger.warn("No suitable negative training position found for positive training example {}. Skipping.".format(pos))
                else:
                    neg_positions.append(neg_pos)
    else:
        # Samples randomly on the chromosome. Note that this procedure can lead to duplicate positions
        while len(neg_positions) < neg_to_pos_training * len(positions):
            new_neg_positions = [x for x in
                                 random.sample(xrange(window_size, max_pos),
                                               neg_to_pos_training * len(positions) - len(neg_positions))
                                 if x not in positions and not in_intervals(ambiguous_bases, x)]
            neg_positions.extend(new_neg_positions)

    return neg_positions


def make_indel_model(window_size):
    indel = Input(shape=(2 * window_size + 1, len(BASES)), name="indel")
    coverage = Input(shape=(2 * window_size + 1, 1), name="coverage")
    x = Conv1D(filters=200, kernel_size=12, activation="relu", kernel_initializer='glorot_normal')(indel)
    x = SpatialDropout1D(0.2)(x)
    x = Conv1D(filters=200, kernel_size=12, activation="relu", kernel_initializer='glorot_normal')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=100, kernel_size=12, activation="relu", kernel_initializer='glorot_normal')(x)
    x = Dropout(0.2)(x)
    y = Conv1D(filters=50, kernel_size=34, activation="relu", kernel_initializer='glorot_normal')(coverage)
    y = SpatialDropout1D(0.2)(y)
    x = concatenate([x,y])
    x = Conv1D(filters=100, kernel_size=12, activation="relu", kernel_initializer='glorot_normal')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=64, kernel_initializer='normal', activation='relu')(x)
    prob_output = Dense(units=2, kernel_initializer='normal', activation='softmax')(x)

    model = Model(inputs=[indel, coverage], outputs=[prob_output])

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.5)
    adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)

    model.compile(optimizer=adamo, loss='binary_crossentropy', metrics=['accuracy'])

    logger.info('model summary:\n{}'.format(model.summary()))

    return model


def bed_file_to_dict(bed_file):
    ''' Create a dict to store intervals from a bed file.

    Arguments:
        bed_file: the file to load

    Returns:
        bed: dict where keys in the dict are contig ids
            values are a tuple of arrays the first array
            in the tuple contains the start positions
            the second array contains the end positions.
    '''
    bed = {}

    with open(bed_file)as f:
        for line in f:
            parts = line.split()
            contig = parts[0]
            lower = int(parts[1])
            upper = int(parts[2])

            if contig not in bed.keys():
                bed[contig] = ([], [])

            bed[contig][0].append(lower)
            bed[contig][1].append(upper)

    for k in bed.keys():
        bed[k] = (np.array(bed[k][0]), np.array(bed[k][1]))

    return bed


def in_bed_file(bed_dict, contig, pos):
    # Exclusive
    lows = bed_dict[contig][0]
    ups = bed_dict[contig][1]
    return np.any((lows < pos) & (pos < ups))


def split_data(datasets, subset_ratios, sequential=False):
    """

    Splits datasets in len(subset_ratios) subsets with ratios as specified in subset_ratios.
    If ratios don't sum up to 1, they are normalized first.

    :param list of array datasets: input datasets
    :param list of float subset_ratios: list of subsets with their respective ratios
    :param bool sequential: If set, the data is partitioned sequentially rather than shuffled
    :return: Subsets
    :rtype: list of array
    """

    samples = datasets[0].shape[0]
    sum_ratios = sum(subset_ratios)
    results_length = [int(x / sum_ratios * samples) for x in subset_ratios]

    indices = range(samples)

    if not sequential:
        random.shuffle(indices)

    results = []
    i = 0
    for n in results_length:
        results.append([x[indices[i:n+i]] for x in datasets])
        i += n

    return results


def plot_metric_history(history, output, title):
    """
    Plots metrics history as pdf.

    :param history:
    :param str output: output file name
    :param str title: plot title
    :return:
    """
    # list all data in history
    print(history.history.keys())

    row = 0
    col = 0
    num_plots = len(history.history) / 2.0  # valid and train plot together
    rows = 4
    cols = max(2, int(math.ceil(num_plots / float(rows))))

    f, axes = plt.subplots(rows, cols, sharex=True, figsize=(36, 24))
    for k in history.history.keys():

        if 'val' not in k:

            axes[row, col].plot(history.history[k])
            axes[row, col].plot(history.history['val_' + k])

            axes[row, col].set_ylabel(str(k))
            axes[row, col].legend(['train', 'valid'], loc='upper left')
            axes[row, col].set_xlabel('epoch')

            row += 1
            if row == rows:
                row = 0
                col += 1
                if row * col >= rows * cols:
                    break

    axes[0, 1].set_title(title)
    plt.savefig(output)


def plot_binned_eval(results, output):
    fit = smf.ols("n_vcf_indels ~ n_pred_indels", data=results).fit()
    plot = sns.regplot('n_vcf_indels', 'n_pred_indels', data=results)
    plot.set_title('regression p-value: {}'.format(fit.pvalues[1]))
    plt.savefig(output)



def get_callbacks(output_prefix, patience=2, batch_size=32):
    checkpointer = ModelCheckpoint(filepath=output_prefix + ".callbacks", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
    tensorboard = TensorBoard(log_dir=output_prefix + "/tensorboard", histogram_freq=1, batch_size=batch_size)
    return [checkpointer, earlystopper, tensorboard]


def tensor_to_str(tensor):
    return "".join(dna_encoder.inverse_transform(tensor))


def get_best_scoring_training_examples(model, training_data, n_best = 50):

    probs = model.predict(training_data[0:2], verbose=1)[:,0]

    # Best scoring positive training examples
    best_scoring_indices = np.argpartition(probs,-n_best)[-n_best:]
    positive_examples = zip(
        [tensor_to_str(t) for t in training_data[0][best_scoring_indices]],
        [np.mean(x) for x in training_data[1][best_scoring_indices]],
        probs[best_scoring_indices].tolist(),
        training_data[2][best_scoring_indices],
        training_data[3][best_scoring_indices])

    # Best scoring negative training examples
    best_scoring_indices = np.argpartition(probs,n_best)[:n_best]
    negative_examples = zip(
        [tensor_to_str(t) for t in training_data[0][best_scoring_indices]],
        [np.mean(x) for x in training_data[1][best_scoring_indices]],
        probs[best_scoring_indices].tolist(),
        training_data[2][best_scoring_indices],
        training_data[3][best_scoring_indices]
    )

    return positive_examples, negative_examples


def apply_model(model, reference_contig, ambiguous_bases_contig, coverage_contig, window_size):
    logger.info("Computing predictions for {} bases.".format(len(reference_contig)))
    start = timer()
    intervals = get_interval_overlaps(ambiguous_bases_contig, window_size, len(reference_contig) - window_size)
    predictions = np.zeros(window_size, dtype=np.float16)
    for interval_start, interval_stop, overlaps in intervals:
        if overlaps:
            predictions = np.append(predictions, np.zeros(interval_stop - interval_start, dtype=np.float16))
        else:
            tensors_bases = np.asarray([ reference_contig[i - window_size -1: i + window_size] for
                        i in range(interval_start, interval_stop)]) #This will have N context around start/end of non-amibugous sequences -- OK?
            tensors_coverage = np.asarray([coverage_contig[i - window_size - 1: i + window_size] for
                                        i in range(interval_start,
                                                   interval_stop)])  # This will have N context around start/end of non-amibugous sequences -- OK?
            predictions = np.append(predictions, model.predict([tensors_bases, tensors_coverage], verbose=1)[:,0].astype(np.float16))

    predictions = np.append(predictions, np.zeros(window_size, dtype=np.float16))
    logger.info("Computed {} predictions in {}.".format(len(predictions), timer() - start))
    return predictions


def eval_predictions(predictions, indels, indel_contigs, ambiguous_bases, bin_size, drop_bins_with_ambiguous=True):

    if not set(indel_contigs.keys()).intersection(set(predictions.keys())):
        logger.error("No contig found in both predictions and indels VCF. Cannot evaluate predictions.")
        return None

    logger.info("Evaluating predictions at ambiguous bases")
    n = [0,0]
    for contig, contig_predictions in predictions.iteritems():
        if contig in ambiguous_bases:
            for interval in ambiguous_bases[contig]:
                if interval[0] > len(contig_predictions): #Mostly for testing when not computing full chromosomes
                    break
                n[0] += np.count_nonzero(contig_predictions[interval[0]:interval[1]])
                n[1] += len(contig_predictions[interval[0]:interval[1]])

    logger.info("Found {}/{} ambiguous bases with a prediction > 0.0".format(*n))

    logger.info("Computing confusion matrix")
    true_positives = np.empty(0,dtype=np.int8)
    binarized_predictions = np.empty(0,dtype=np.int8)

    for contig, (start_index, end_index) in indel_contigs.iteritems():
        if contig in predictions:
            indel_index = start_index
            contig_length = len(predictions[contig])
            intervals = get_interval_overlaps(ambiguous_bases[contig], 0, contig_length) if contig in ambiguous_bases else [dict(start=0, end=contig_length, overlaps=False)]

            for interval_start, interval_stop, overlaps in intervals:
                interval_indels = [x.POS - interval_start - 1 for x in indels[indel_index:end_index] if x.POS <= interval_stop]
                if not overlaps:
                    interval_tps = np.zeros(interval_stop - interval_start, dtype=np.int8)
                    interval_tps[interval_indels] = 1 #If stmt is for testing when not computing full chromosomes
                    true_positives = np.append(true_positives, interval_tps)

                    interval_binarized_predictions = np.zeros(interval_stop - interval_start, dtype=np.int8)
                    interval_binarized_predictions[predictions[contig][interval_start:interval_stop] > 0.5] = 1
                    binarized_predictions = np.append(binarized_predictions, interval_binarized_predictions)
                indel_index += len(interval_indels)
    pprint(
        pd.DataFrame(confusion_matrix(true_positives, binarized_predictions),
                     columns=['pred_neg','pred_indel'],
                     index=['truth_neg','truth_indel'])
           )

    logger.info("Computing predictions / observation counts in bins of size {}".format(bin_size))

    results = []
    for contig, (start_index, end_index) in indel_contigs.iteritems():
        if contig in predictions:
            bins = [(i, i+bin_size) for i in range(0, len(predictions[contig]), bin_size)]
            indel_index = start_index
            for start_pos, end_pos in bins:
                n_vcf_indels = sum(1 for x in
                                   itertools.takewhile(lambda x: x.POS < end_pos, # Strictly smaller since VCF is 1-based
                                                       indels[indel_index:end_index]))

                n_pred_indels = np.sum(predictions[contig][start_pos:end_pos])
                results.append((
                    contig,
                    start_pos + 1,
                    end_pos,
                    n_vcf_indels,
                    n_pred_indels,
                    contig in ambiguous_bases and overlaps_intervals(ambiguous_bases[contig], start_pos, end_pos)
                ))
                indel_index += n_vcf_indels

    results = pd.DataFrame.from_records(results, columns=['chrom','start','end','n_vcf_indels','n_pred_indels','overlaps_ambiguous'])

    if drop_bins_with_ambiguous:
        results = results[results.overlaps_ambiguous == False]

    pprint(results)

    logger.info("Obs/Exp Pearson correlation: {}".format(results[['n_vcf_indels','n_pred_indels']].corr(method='pearson')['n_vcf_indels']['n_pred_indels']))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', help='One-hot-encoded reference (required if not using --ref_fasta)', required=False)
    parser.add_argument('--ref_fasta', help='Reference fasta. If used, the reference will be one-hot-encoded and saved as a npy files. (required if not using --ref)', required=False)
    parser.add_argument('--coverage', help='Normalized coverage npy files location', required=False)
    parser.add_argument('--train_vcf', help='VCF(s) containing indel training examples. Required to train a model.', required=False, nargs="+")
    parser.add_argument('--eval_vcf', help='VCF(s) containing indel evaluation examples. If not specified, VCF used for training is used. Either --train_vcf or --eval_vcf is required for evaluation.', required=False, nargs="+")
    parser.add_argument('--output', help='Output prefix', required=True)
    parser.add_argument('--window_size', help='Size of the window to consider on each side of the indel (default 30).',
                        required=False, default=30, type=int)
    parser.add_argument('--max_training',
                        help='Maximum number of training examples to use for each class (default 100,000). Set to -1 to use all.',
                        required=False, default=100000, type=int)
    parser.add_argument('--neg_train_window',
                        help='Size of the window to consider on each side of the indel to sample a negative training example (default 1000). If set to a value < 1 => select a random position on the same chromosome.',
                        required=False, default=1000, type=int)
    parser.add_argument('--neg_to_pos_training',
                        help='Number of negative training examples per positive training example (default 1).',
                        required=False, default=1, type=int)
    parser.add_argument('--pass_only', help='Use PASS indels only.',
                        action='store_true', required=False)
    parser.add_argument('--train_model', help='Trains a new model. Required if not using --callback.', required=False,
                        action='store_true')
    parser.add_argument('--callback', help='When specified, will initialize the model weights with the given callback.', required=False)
    parser.add_argument('--save_best_examples', help='When specified, saves the best training examples (for each class).',
                        action='store_true', required=False)
    parser.add_argument('--compute_predictions', help='Computes predictions for every base in the genome and writes them to disk.',
                        action='store_true', required=False)
    parser.add_argument('--load_predictions', help='Loads predictions for every base in the genome from the directory specified.',
                        required=False)
    parser.add_argument('--eval_predictions', help='Evaluate predictions against indels.',
                        action='store_true', required=False)
    parser.add_argument('--eval_bin_size',
                        help='Size of the bins to compute obs and exp (default 5,000). Needs to be >0.',
                        required=False, default=5000, type=int)
    parser.add_argument('--predictions_interval',
                        help='Limit predictions to the start of a single contig, in the form contig:end.',
                        required=False)

    args = parser.parse_args()

    if int(args.ref_fasta is not None) + int(args.ref is not None) != 1:
        sys.exit("One and only one of --ref_fasta or --ref is required.")

    if not args.train_model and not args.callback:
        sys.exit("Must specify --callback when not using --train_model.")

    if args.compute_predictions and args.load_predictions:
        sys.exit("Only one of --compute_predictions and --load_predictions can be specified.")

    if (args.train_model or args.save_best_examples) and (args.train_vcf is None):
        sys.exit("--train_vcf needs to be specified to train a model (--train_model) or save best training examples (--save_best_examples).")

    if args.eval_predictions and args.train_vcf is None and args.eval_vcf is None:
        sys.exit("At least one of --train_vcf or --eval_vcf needed for evaluation of predictions (--eval_predictions).")

    if args.eval_bin_size < 1:
        sys.exit("--eval_bin_size value must be > 0.")

    main(args)
