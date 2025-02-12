import os
import subprocess
import numpy as np
import whisper
import torch
import argparse
from vad import EnergyVAD


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default= "./a_dataset/spanish/", help="data wav dir")
parser.add_argument("--model_type", type = str, default="tiny", help="model type")
parser.add_argument("--run_device", type = str, default="cpu", help="run device")
parser.add_argument("--window_length", type = float, default= 0.2, help="window length")
parser.add_argument("--overlap", type = float, default= 0, help="overlap")


# Define
opt = parser.parse_args()
model_type = opt.model_type
run_device = opt.run_device
window_length = opt.window_length
overlap = opt.overlap
DATA_DIR = opt.data_dir + f"/samples"
SEGMENT_DIR = opt.data_dir + f"/segments_{window_length}"

if not os.path.exists(SEGMENT_DIR):
    os.makedirs(SEGMENT_DIR)

# Load model
whisper_model = whisper.load_model(model_type, run_device)
# embedding_dims = {"tiny": 384, 'small': 768, 'base': 512, 'medium':1024, 'large': 1280}




def extract_segment(input_file, output_file,  start_time, end_time):
    """
    Extract one segment given start_time and end_time
    input_file: input .wav file
    output_file: extracted .wav segment
    start_time, end_time: start-end time of the segment
    """
    # split_file_name = f'./{input_file}_segment_{start_time}_{end_time}.wav'
    cmd= 'ffmpeg -i '+input_file+' -acodec copy -ss '+str(start_time)+' -to '+str(end_time)+' '+ output_file
    os.system(cmd)



def split_audio_with_ffmpeg(input_file, output_dir, segment_length=window_length, overlap=overlap):
    """
    Extract all segments from original audio
    """
    input_filename = input_file.split("/")[-1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    duration = float(subprocess.check_output(['ffprobe', '-i', input_file, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")]))

    start_time = 0
    last_flag = False 
    while start_time < duration and last_flag == False:
        end_time = np.round(min(start_time + segment_length, duration), 2)
        # Cover the last segment
        if end_time + segment_length > duration:
            end_time = duration
            last_flag = True
        output_file = os.path.join(output_dir, f"{input_filename}_segment_{start_time}_{end_time}.wav")
        extract_segment(input_file, output_file, start_time, end_time)
        start_time += segment_length - overlap



def extract_segment_embedding(segment_dir, save_segment_dir, window_length):
    """
    Extract embedding for each segment
    """

    audio = whisper.load_audio(segment_dir)
    # load audio file in "audio" variable

    vad = EnergyVAD(
        sample_rate= 16000,
        frame_length = 25, # in milliseconds
        frame_shift = 20, # in milliseconds
        energy_threshold = 0.05, 
        pre_emphasis = 0.95,
    ) # default values are used here

    voice_activity = vad(audio) # returns a boolean array indicating whether a frame is speech or not
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)    
    #--- this code to create the correct shape of mel spectrogram
    while True:
        nF, nT = np.shape(mel)
        # print(nF, nT)
        if nT > 3000:
            mel = mel[:,0:3000]
            break
        else:
            mel = torch.cat((mel, mel), -1)
    mel = torch.unsqueeze(mel, 0) 
    
    wp_emb = whisper_model.embed_audio(mel)
    # print("Wb_emb shape:", wp_emb.shape)
   
    emb_1d  = np.mean(wp_emb.cpu().detach().numpy(), axis=0)

    emb_1d  = np.mean(emb_1d, axis=0)

    emb_1d = np.expand_dims(emb_1d, axis = 0)
   
    if not voice_activity[0]:
        return np.zeros_like(emb_1d)
    
    np.save(save_segment_dir + '/{}.npy'.format(segment_dir.split("/")[-1]), emb_1d, allow_pickle=True)

    return emb_1d 



def delete_segment_after_done(segments_dir):
    """
    Delete segment after extracting embedding
    """
    for segment in os.listdir(segments_dir):
        if segment.endswith('.wav'):
            segment_path = segments_dir + "/" + segment
            cmd = 'rm '+ segment_path
            os.system(cmd)




if __name__ == "__main__":
    data_list = sorted(os.listdir(DATA_DIR))

    for sample in data_list:
        sample_path = DATA_DIR + "/" + sample
        segment_save_dir = SEGMENT_DIR + "/" + sample
        print("Sample path", sample_path)
        if not os.path.exists(segment_save_dir):
            os.makedirs(segment_save_dir)
        # # Extract segments    
        split_audio_with_ffmpeg(input_file=sample_path, output_dir=segment_save_dir)

        # Extract embedding
        segment_list = sorted(os.listdir(segment_save_dir), key= lambda x: x.split("_")[-2])
        for segment in segment_list:
            segment_path = segment_save_dir + "/" + segment
            # Extract embedding each segment
            embed_1d = extract_segment_embedding(segment_dir=segment_path, save_segment_dir= segment_save_dir,window_length=window_length)
        
        # Delele segment wav after embeddings are extracted
        delete_segment_after_done(segments_dir=segment_save_dir)
    

    
 