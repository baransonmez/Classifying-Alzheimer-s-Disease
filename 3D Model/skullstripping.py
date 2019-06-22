import os
import nipype
import nipype.interfaces.fsl as fsl

data_dir='C:/Users/merve/Desktop/INFILES/Train/AD'
# path to raw image directory
ssdata_dir='C:/Users/merve/Desktop/INFILES/Train/Str'
# path to skull stripped image directory


def strip():
    for file in os.listdir(data_dir):
        # strip with frac = 0.5
        mybet = nipype.interfaces.fsl.BET(in_file=os.path.join(data_dir,file),out_file=os.path.join(ssdata_dir,file +'_stripped.nii'), frac=0.5)                #frac=0.2
        mybet.run()
        print(file+' is skull stripped')
