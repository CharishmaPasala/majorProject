import torch,librosa,pickle
import numpy as np

class ML_model:
    def __init__(self):
        self.ml_model=torch.load(r"support_file\resnet_carcrash_94.pth",map_location=torch.device('cpu'))
        self.ml_model.eval()
        with open(r'support_file\indtocat.pkl','rb') as f:
            self.i2c=pickle.load(f)


    # def pre_process(self,input_image):

    def spec_to_image(self,spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled



    def get_melspectrogram_db(self,file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
        wav,sr = librosa.load(file_path,sr=sr)
        sr= 44100
        if wav.shape[0]<5*sr:
            wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
        else:
            wav=wav[:5*sr]
        spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
        spec_db=librosa.power_to_db(spec,top_db=top_db)

        return spec_db
    
    def get_prediction(self,file_path):
        #input_image=self.spec_to_image(self.get_melspectrogram_db(file_path))[np.newaxis,...]
        # output_pred=self.ml_model(input_image)
        input_image=self.spec_to_image(self.get_melspectrogram_db(file_path))
        spec_t=torch.tensor(input_image).to("cpu", dtype=torch.float32)
        pr=self.ml_model.forward(spec_t.reshape(1,1,*spec_t.shape))
        ind = pr.argmax(dim=1).cpu().detach().numpy().ravel()[0]
        return self.i2c[ind]


    
