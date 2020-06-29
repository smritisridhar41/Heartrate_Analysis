import csv
import sys
from pathlib import Path
import numpy as np
import scipy as sp
from datetime import datetime
from scipy import signal
from sklearn import svm
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
sys.path.append('libs')
import detect_peaks

class HRClassifier:

    sampleWindow = 500#1600 # PPG readings for feature extraction

    def train(self, trainAnyway = False):
        if not trainAnyway:
            # Check that trained classifier doesn't already exit
            my_file = Path('trainedHRclassifier.pkl')
            if my_file.is_file():
                return

        # Extract features from known valid and invalid segments
        HRfeatures = self.extractAllFeatures(self.HRranges, isValidHR = True)
        NonHRfeatures = self.extractAllFeatures(self.NonHRranges, isValidHR = False)
        allFeatures = HRfeatures + NonHRfeatures

        # Label valids as 1 and invalids as 0
        allLabels = [1 for _ in range(len(HRfeatures))] + [0 for _ in range(len(NonHRfeatures))] 
        
        # Train, leave out 20% of features
        allFeaturesTrain, allFeaturesTest, allLabelsTrain, allLabelsTest = train_test_split(allFeatures, allLabels, test_size=0.2)
        
        # Use a linear model classifier
        model = svm.SVC(kernel='linear', probability=True)
        model.fit(allFeaturesTrain, allLabelsTrain) 

        # Save the trained classifier for future use
        joblib.dump(model, 'trainedHRclassifier.pkl') 

        # Accuracy (0-1 value)
        print(model.score(allFeaturesTest, allLabelsTest))
        predictedLabels = model.predict(allFeaturesTest)

    # Params: 1) Nx2 array of PPG readings and corresponding timestamps
    #          2) For training, whether the array readings represent a valid heart rate segment
    #          3) For test, whether to include heart rate number in feature set
    # Return: an array of feature vectors, each from a sampleWindow subset of the input set
    # See further explanation above
    def extractFeatures(self, PPGdata, isValidHR = False, returnHR = False):
        PPGsize, columnsSize = PPGdata.shape

        features = []

        for i in range(PPGsize//self.sampleWindow):
            start = self.sampleWindow*i
            end = self.sampleWindow*(i+1)-1

            if PPGdata[start, 1] == 0 or PPGdata[end, 1] == 0:
                continue

            dataWindow = PPGdata[start:end, 0]

            # Filter outliers using a moving median filter
            dataWindow = signal.medfilt(dataWindow, 3)
            
            # Power spectral density (self.sampleWindow/4 reading window, self.sampleWindow/(4*2) sequential window overlap)
            f, pxx = signal.welch(dataWindow, fs = 90, nperseg = self.sampleWindow/4)
            pxx = [10*np.log10(x) for x in pxx] 

            # Mean amplitude of HF components (20 to 200Hz range)
            m = np.mean(pxx[20:200])

            # LF/HF component magnitude ratio
            lfhfratio = (pxx[1] + pxx[2]) / (pxx[4] + pxx[5])

            # Apply a 2nd order bandpass butterworth filter to attenuate heart rate components
            # other than heart signal's RR peaks.
            b, a = signal.butter(2, [1/100, 1/10], 'bandpass')
            dataWindow = signal.filtfilt(b, a, dataWindow)
            # Find the segment peaks' indices for peak occurrence variance feature.
            indices = detect_peaks.detect_peaks(dataWindow, mph = 30, mpd = 20)
            peakVariance = np.finfo(np.float64).max
            if len(indices) > 1:
                peakVariance = np.var(np.diff(indices))

            # Calculate the heart rate number from number of peaks and start:end timeframe 
            startTime = self.getReadingTime(PPGdata, start)
            endTime = self.getReadingTime(PPGdata, end)
            timeDifference = endTime - startTime
            timeDifferenceInMinutes = float(timeDifference.seconds + float(timeDifference.microseconds)/10**6)/60.0
            heartRate = float(len(indices)) / timeDifferenceInMinutes

            # Filter out HR values outside of normal human range (50-120Hz), if this is a known valid HR segment
            if isValidHR == False or (isValidHR == True and heartRate < 120 and heartRate > 50):
                if returnHR == False:
                    features.append([m, lfhfratio, peakVariance])
                else:
                    features.append([m, lfhfratio, peakVariance, heartRate])
        return features


    # See extractFeatures method
    def extractAllFeatures(self, allRanges, isValidHR = False, returnHR = False):
        featureSetP = []
        for date in allRanges:
            print(date)
            # Load training set PPG record file
            PPGdata = self.getPPGDataFromFile(date)
            PPGdata[0,:] = self.correctSaturation(PPGdata[0,:])

            for r in allRanges[date]:
                featureSetP += self.extractFeatures(PPGdata[r[0]:r[1], :], isValidHR = isValidHR, returnHR = returnHR)
        return featureSetP


    # Params: test set Nx2 array of PPG readings and corresponding timestamps
    # ReturnL list of valid heart rate ranges and the calculated rates
    def getValidHRranges(self, PPGdata):
        try:
            model = joblib.load('trainedHRclassifier.pkl') 
        except:
            print("Trained classifier not found, need to train first.")
            raise

        features = np.array(self.extractFeatures(PPGdata, isValidHR = True, returnHR = True))

        validHRranges = []

        labels = model.predict(features[:,0:3])
        ds = np.absolute(np.array(model.decision_function(features[:,0:3])))

        for i,f in enumerate(features):
            label = labels[i]
            confidence = ds[i]

            if confidence < 0.3 and label == 1:
                validHRranges.append([int(f[3]), confidence])

        return validHRranges


    # Correct signal saturation (defined as sharp slope in curve) 
    # by scaling subsequent readings by the maximum range
    # Params, Return: list of PPG readings (floats)
    def correctSaturation(self, signal):
        maxval = max(signal);
        signalD = np.diff(signal);
        signalDEnd = len(signalD)-1;
        maxSlope = 50000;

        for k in range(len(signalD)):
            if signalD[k] > maxSlope:
                # Pull subsequent values down
                signal[k+1:signalDEnd] = signal[k+1:signalDEnd] - maxval;
            if signalD[k] < -1*maxSlope:
                # Pull subsequent values up
                signal[k+1:signalDEnd] = signal[k+1:signalDEnd] + maxval;
        return signal


    # Params: string in format month_day_deviceID
    # Return: 2xN array of PPG readings and their timestamps
    def getPPGDataFromFile(self, dateDeviceID):
        return np.array(list(csv.reader(open("dataPPG/data_%s_PPG.csv" % dateDeviceID, "rt"), delimiter=','))).astype('float')

    def getPPGDataFromStream(self, source):
        data = []
        i = 0
        while i<self.sampleWindow:
            data.append(source.readline().strip().decode("utf-8").split(','))
            i+=1
        return np.array(data).astype('float')

    def getReadingTime(self, PPGdata, readingID):
        return datetime(int(PPGdata[readingID, 1]), int(PPGdata[readingID, 2]), int(PPGdata[readingID, 3]), 
                        int(PPGdata[readingID, 4]), int(PPGdata[readingID, 5]), int(PPGdata[readingID, 6]), 
                        int((PPGdata[readingID, 6] % 1)*(10**6)))


    # Manually observed ranges in (start,end) format of valid and invalid PPG readings for 
    # different devices and days during a single week, used for feature extraction in self.train 
    HRranges = {
    #'6_2_1': [[61911,64962],[65446,67882],[80208,85046],[88139,90207],[223169,223899],[232908,234236],[265489,267293],[353947,354697],[366813,371476],[411791,415532],[448336,449363]],
    '6_2_2': [[203981,208131],[215986,217841],[221827,222478],[437989,439886],[458251,459258],[493759,494177]],
    '6_2_3': [[221380,222809]],
    '6_1_1': [[391997,395445],[396675,400513],[411186,415921],[425726,426439],[430232,432188],[439971,441243],[491068,492569]],
    '6_1_2': [[143511,143975],[300087,301138],[392455,392983],[398491,399516],[400320,401172],[402684,403640],[403854,407949],[408882,412079],[415804,416261],[436978,437527],[443985,444455],[445525,446376]],
    '6_1_3': [[113552,115794]],
    '6_1_4': [[55912,56500],[56991,58063],[61791,62657],[67132,67983],[68948,70386],[70585,71657],[73099,74212],[79848,80867],[86573,89705],[94715,96642],[96735,102051],[105413,106107],[108072,108808],[109292,118669],[126105,126586],[131356,133058],[133747,134458],[135640,136572],[137681,139264],[140290,143900],[146079,147153],[148802,152134],[154275,155777],[246174,263619],[278957,285305],[286130,289059],[295653,302292],[307965,310724],[361960,363442],[366771,367844],[379721,381872],[384094,397889],[401916,402939],[404748,408671],[409326,415055],[416749,425676],[427470,442650],[455548,456752],[466106,467333],[468511,469468],[469943,470837],[494808,497432],[580152,589637],[602129,602786],[603287,604102],[622554,623635],[624297,626239],[631506,633657],[652580,653997],[666283,667131],[667721,668139]]
    }
    NonHRranges = {
    '6_1_2': [[8761,136626],[153858,293883],[312192,383931],[451405,645131]]
    }
