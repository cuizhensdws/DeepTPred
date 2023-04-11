
'''
from __future__ import print_function # enable Python3 printing
import numpy

import matplotlib.pyplot as plt
from keras.models import Model
from keras import initializers
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.utils import shuffle
from Bio import SeqIO
import pandas as pd
import scipy.io as scio
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Layer, InputSpec
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras_self_attention import SeqSelfAttention


#User-set variables
max_length = 100
amp_train_fasta       = 'Datasets_V2/AMP.tr.fa'
#amp_validate_fasta    = 'Datasets_V2/AMP.eval.fa'
amp_test_fasta        = 'Datasets_V2/AMP.te.fa'
decoy_train_fasta     = 'Datasets_V2/Decoy.tr.fa'
#decoy_validate_fasta  = 'Datasets_V2/Decoy.eval.fa'
decoy_test_fasta      = 'Datasets_V2/Decoy.te.fa'


#Model params
embedding_vector_length = 60
nbf = 20		# No. Conv Filters
flen = 20 		# Conv Filter length+
nlstm = 20	    # No. LSTM layers
ndrop = 0.2    # LSTM layer dropout
nbatch = 40 	# Fit batch No.
nepochs = 60   # No. training rounds

amino_acids = "XBJZACDEFGHIKLMNPQRSTVWY"
aa2int = dict((c, i) for i, c in enumerate(amino_acids))

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

print("Encoding training/testing sequences...")
for s in SeqIO.parse(amp_train_fasta,"fasta"):
    X_train.append([aa2int[aa] for aa in str(s.seq).upper()])
    y_train.append(1)
#for s in SeqIO.parse(amp_validate_fasta,"fasta"):
    #X_val.append([aa2int[aa] for aa in str(s.seq).upper()])
    #y_val.append(1)
for s in SeqIO.parse(amp_test_fasta,"fasta"):
    X_test.append([aa2int[aa] for aa in str(s.seq).upper()])
    y_test.append(1)
for s in SeqIO.parse(decoy_train_fasta,"fasta"):
    X_train.append([aa2int[aa] for aa in str(s.seq).upper()])
    y_train.append(0)
#for s in SeqIO.parse(decoy_validate_fasta,"fasta"):
    #X_val.append([aa2int[aa] for aa in str(s.seq).upper()])
    #y_val.append(0)
for s in SeqIO.parse(decoy_test_fasta,"fasta"):
    X_test.append([aa2int[aa] for aa in str(s.seq).upper()])
    y_test.append(0)

# Pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
#X_val = sequence.pad_sequences(X_val, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)

#dataFile = 'Data0.mat'
#data = scio.loadmat(dataFile)
#X1 = data.get('X1')
#X2 = data.get('X2')
#X_train = X1.astype(numpy.float64)
#X_test  = X2.astype(numpy.float64)



#data1 = pd.DataFrame(X_train, index=None, columns=None)
#data1.to_csv('X_train_SBP.csv')
#save = pd.DataFrame(data1, )
#data2 = pd.DataFrame(X_test, index=None, columns=None)
#data2.to_csv('X_test_SBP.csv')
#save = pd.DataFrame(data2, )


# Shuffle training sequences


X_train, y_train = shuffle(X_train, numpy.array(y_train))
X_val, y_val = shuffle(X_val, numpy.array(y_val))

#gljz1 = numpy.array(X_train)
#data1 = pd.DataFrame(gljz1, index=None, columns = None)
#gljz2 = numpy.array(X_val)
#data2 = pd.DataFrame(gljz2, index=None, columns = None)
#gljz3 = numpy.array(X_test)
#data3 = pd.DataFrame(gljz3, index=None, columns = None)


#data1.to_csv('gljz1.csv')
#data2.to_csv('gljz2.csv')
#data3.to_csv('gljz3.csv')





print("Compiling model...")
model = Sequential()
model.add(Embedding(24, embedding_vector_length, input_length=max_length,name='emb'))
#model.add(SeqSelfAttention(attention_activation='sigmoid',attention_type='multiplicative',name='SA'))
model.add(Conv1D(filters=nbf, kernel_size=flen, padding="same", activation='relu',name='conv'))

model.add(MaxPooling1D(pool_size=5,name='mp'))

model.add(LSTM(nlstm, use_bias=True, dropout=ndrop, return_sequences=False,name='lstm'))#,merge_mode='ave'))

model.add(Dense(1, activation='sigmoid',name='ds'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training now...")
model.fit(X_train, numpy.array(y_train), epochs=nepochs, batch_size=nbatch, verbose=1)

middle = Model(inputs=model.input,outputs=model.get_layer('lstm').output)
result = middle.predict(X_test)

print(result.shape)
#np.reshape(A,(-1,2))
Result=numpy.savetxt('result.csv',result)
#plt.figure()
#plt.scatter(result[:,0],result[:,1])
#plt.show()













print("\nGathering Testing Results...")
preds = model.predict(X_test)
pred_class = numpy.rint(preds) #round up or down at 0.5
true_class = numpy.array(y_test)
tn, fp, fn, tp = confusion_matrix(true_class,pred_class).ravel()
roc = roc_auc_score(true_class,preds)
mcc = matthews_corrcoef(true_class,pred_class)
acc = (tp + tn) / (tn + fp + fn + tp + 0.0)
sens = tp / (tp + fn + 0.0)
spec = tn / (tn + fp + 0.0)
prec = tp / (tp + fp + 0.0)
#f1 = (2*tp) / (2*tp + fp + fn + 0.0)

print("\nTP\tTN\tFP\tFN\tSens\tSpec\tAcc\tMCC\tPrec\tauROC")
print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".
      format(tp,tn,fp,fn,numpy.round(sens,4),numpy.round(spec,4),numpy.round(acc,4),numpy.round(mcc,4),numpy.round(prec,4),numpy.round(roc,4)))

# END PROGRAM




