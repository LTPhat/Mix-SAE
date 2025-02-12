import numpy as np
import pandas as pd
import argparse
import os
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment, Timeline, Annotation





def createDER(label_path, sample_dir, prediction, window_length, overlap):
    """
    Extract series from label and prediction for calculating DER
    """

    column = ['speaker','file_name','number' ,'start', 'duration', 'na1', 'na2', 'label', 'na3', 'na4']
    df = pd.read_csv(label_path, delimiter=' ', header=None, usecols=column, names=column)
    refer = Annotation(uri='label')

    # Assign label
    prev_end = 0
    for row in df.iterrows():
        row_item = row[1]
        start = np.round(row_item['start'], 2)
        end = np.round(row_item['start'] + row_item['duration'], 2)
        # Avoid overlap
        if start < prev_end:
            start = prev_end
        # Avoid error label
        if start > end:
            continue
        refer[Segment(start, end)] = row_item['label']
        prev_end = end
    print("******EXTRACT LABEL DONE***********")

    # assert len(os.listdir(sample_dir)) == len(prediction)
    segment_list = sorted(os.listdir(sample_dir), key= lambda x: float(x.split("_")[-2]))

    # Create index mapping to store start-end index of consecutive segments
    index_mapping = {}
    start_index = 0
    current_value = prediction[0]
    
    for i in range(1, len(prediction)):
        if prediction[i] != current_value:
            index_mapping[(start_index, i - 1)] = current_value
            start_index = i
            current_value = prediction[i]
    
    # Handle the last consecutive sequence
    index_mapping[(start_index, len(prediction) - 1)] = current_value
    

    # Assign label to consecutive segments
    hyp = []
    for key, value in index_mapping.items():
        start_index = key[0]
        end_index = key[1]
        speaker_label = "spk0{}".format(value)
        if overlap != 0:
            start_time = np.round(overlap * start_index, 2)
            if start_index == end_index:
                end_time = np.round(start_time + window_length, 2)
            else:
                end_time = np.round(overlap * end_index + window_length, 2)
        # Non-overlap
        else:
            start_time = np.round(window_length * start_index, 2)
            if start_index == end_index:
                end_time = np.round(start_time + window_length, 2)
            else:
                end_time = np.round((end_index + 1) * window_length, 2)

        hyp.append((speaker_label, start_time, end_time))


    hypo = Annotation(uri='hypo')
    for item in hyp:
        hypo[Segment(item[1], item[2])] = item[0]

    print("******EXTRACT HYP DONE***********")
    
    return refer, hypo


# def create_DER_pyannote(label_path, pyannote_label_path):
#     # df = pd.read_csv(label_path, delimiter=' ', header=None, usecols=column, names=column)

#     # ref = []
#     # prev_end = 0
#     # # Assign label
#     # for row in df.iterrows():
#     #     row_item = row[1]
#     #     start = np.round(row_item['start'], 2)
#     #     end = np.round(row_item['start'] + row_item['duration'], 2)
#     #     # Avoid overlap
#     #     if start < prev_end:
#     #         start = prev_end
#     #     # Avoid error label
#     #     if start > end:
#     #         continue
#     #     ref.append((row_item['label'], start, end))
#     #     prev_end = end
#     df = pd.read_csv(label_path, delimiter=' ', header=None, usecols=column, names=column)
#     refer = Annotation(uri='label')

#     # Assign label
#     for row in df.iterrows():
#         row_item = row[1]
#         start = np.round(row_item['start'], 2)
#         end = np.round(row_item['start'] + row_item['duration'], 2)
#         # # Avoid overlap
#         # if start < prev_end:
#         #     start = prev_end
#         # # Avoid error label
#         # if start > end:
#         #     continue
#         refer[Segment(start, end)] = row_item['label']
#         # ref.append((row_item['label'], start, end))
#         # prev_end = end

#     print("******EXTRACT LABEL DONE*****c******")

#     df = pd.read_csv(pyannote_label_path, delimiter=' ', header=None, usecols=column, names=column)
#     print(df)
#     pyannote_ref = []
#     prev_end = 0
#     # Assign label
#     for row in df.iterrows():
#         row_item = row[1]
#         start = np.round(row_item['start'], 2)
#         end = np.round(row_item['start'] + row_item['duration'], 2)
#         # Avoid overlap
#         if start < prev_end:
#             start = prev_end
#         # Avoid error label
#         if start > end:
#             continue
#         pyannote_ref.append((row_item['label'], start, end))
#         prev_end = end
#     print("******EXTRACT PYANNOTE LABEL DONE***********")
#     return refer, pyannote_ref


# def create_pyannote_timeline(label_path, pyannote_label_path):
#     df = pd.read_csv(label_path, delimiter=' ', header=None, usecols=column, names=column)
#     refer = Annotation(uri='label')
#     # ref = []
#     # prev_end = 0
#     # Assign label
#     for row in df.iterrows():
#         row_item = row[1]
#         start = np.round(row_item['start'], 2)
#         end = np.round(row_item['start'] + row_item['duration'], 2)
#         # # Avoid overlap
#         # if start < prev_end:
#         #     start = prev_end
#         # # Avoid error label
#         # if start > end:
#         #     continue
#         refer[Segment(start, end)] = row_item['label']
#         # ref.append((row_item['label'], start, end))
#         # prev_end = end

#     print("******EXTRACT LABEL DONE***********")

#     df = pd.read_csv(pyannote_label_path, delimiter=' ', header=None, usecols=column, names=column)
#     py_refer = Annotation(uri='py_label')
#     ref = []
#     # prev_end = 0
#     # Assign label
#     for row in df.iterrows():
#         row_item = row[1]
#         start = np.round(row_item['start'], 2)
#         end = np.round(row_item['start'] + row_item['duration'], 2)
#         # # Avoid overlap
#         # if start < prev_end:
#         #     start = prev_end
#         # # Avoid error label
#         # if start > end:
#         #     continue
#         py_refer[Segment(start, end)] = row_item['label']
#         # ref.append((row_item['label'], start, end))
#         # prev_end = end

#     print("******EXTRACT PY LABEL DONE***********")

#     return refer, py_refer


