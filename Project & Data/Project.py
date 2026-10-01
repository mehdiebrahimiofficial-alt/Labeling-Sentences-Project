import numpy as np
import pandas as pd 
import warnings
import matplotlib.pyplot as plt
import pickle


warnings.filterwarnings('ignore')

## read all data sets and concat to A DATA
amazon=pd.read_csv('amazon_cells_labelled.txt',sep='\t',header=None,names=['Sentences','Labels'])
imbd=pd.read_csv('imdb_labelled.txt',sep='\t',header=None,names=['Sentences','Labels'])
yelp=pd.read_csv('yelp_labelled.txt',sep='\t',header=None,names=['Sentences','Labels'])

Data=pd.concat([amazon,imbd,yelp],ignore_index=True)

## Split Data
from sklearn.model_selection import train_test_split

x=Data['Sentences']
y=Data['Labels']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.2,random_state=42)

## tokenization
import tensorflow as tf
import keras
from keras import layers

tokenizer=layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=100,
    output_mode='int'
)

tokenizer.adapt(x_train)

x_train_token=tokenizer(x_train)
x_val_token=tokenizer(x_val)
x_test_token=tokenizer(x_test)

## positional Encoding
class PositionalEncoder(layers.Layer):
    def __init__(self,max_length,d_model):
        super().__init__()
        self.supports_masking=True

        position=np.arange(max_length)[:,np.newaxis]
        dimention=np.arange(d_model)[np.newaxis,:]
        angel_rates=1/np.power(10000,(2*(dimention//2))/d_model)

        angel_rads=position * angel_rates

        angel_rads[:,0::2]=np.sin(angel_rads[:,0::2])
        angel_rads[:,1::2]=np.cos(angel_rads[:,1::2])

        self.pos_encoding=tf.constant(angel_rads,dtype=tf.float32)

    def call(self,inputs):
        return inputs + self.pos_encoding

## preparing Data for Encoder
input=layers.Input(
    shape=(100,),
    dtype='int32'
)

x=layers.Embedding(
    input_dim=10000,
    output_dim=64,
    mask_zero=True
)(input)

x=PositionalEncoder(
    max_length=100,
    d_model=64
)(x)

## Creat Transformer Model

attention_output=layers.MultiHeadAttention(
    num_heads=4,
    key_dim=16
)(x,x)

x=x + attention_output

x= layers.LayerNormalization()(x)
x= layers.Dropout(0.2)(x)

ffn=layers.Dense(128,activation='relu')(x)
ffn_output=layers.Dense(64)(ffn)

x= x + ffn_output
x = layers.LayerNormalization()(x)
x= layers.Dropout(0.2)(x)

x = layers.GlobalAveragePooling1D()(x)
output=layers.Dense(1,activation='sigmoid')(x)

## Train Model
Model=keras.Model(
    inputs=input,
    outputs=output
)

Model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(Model.summary())

early_stop=keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

Training=Model.fit(x_train_token,y_train,
epochs=10,batch_size=32,validation_data=[x_val_token,y_val],callbacks=[early_stop])

loss,accuracy=Model.evaluate(x_test_token,y_test)
print('LOSS:',loss)
print('ACCURACY:',accuracy)

y_pre=Model.predict(x_test_token)
Prediction = (y_pre >= 0.5).astype(int).flatten()

from sklearn.metrics import classification_report,confusion_matrix

cm=confusion_matrix(y_test,Prediction)

print('Confusion Matrix:',cm)
print('Summary Report:',classification_report(y_test,Prediction))


plt.figure(figsize=(12,8))
plt.plot(Training.history['accuracy'])
plt.plot(Training.history['val_accuracy'])
plt.title('Results Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'])
plt.savefig('Results Accuracy.png')

plt.figure(figsize=(12,8))
plt.plot(Training.history['loss'])
plt.plot(Training.history['val_loss'])
plt.title('Results Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train','Validation'])
plt.savefig('Results Loss.png')

plt.show()


Model.save('sentiment_transformer.keras')

vocabulary=tokenizer.get_vocabulary()
with open('Vocabulary.pkl','wb') as file :
    pickle.dump(vocabulary,file)
    
print('All Files Saved!')

