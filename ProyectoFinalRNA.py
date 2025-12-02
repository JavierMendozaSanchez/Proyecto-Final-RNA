#importamos lo necesario
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
import random
from google.colab import drive, files

#configuramos los datos y valores a usar 
CONFIG = {
    'celeba_total_samples': 4000,   # muestra de celebA
    'img_height': 128,
    'img_width': 128,
    'batch_size': 16,               # empleamos un batch pequeño para generalizar mejor con pocos datos
    'pretrain_epochs': 10,          # epocas para aprender filtros de caras
    'fast_epochs': 25,              # epocas para fase congelada 
    'fine_tune_epochs': 20,         # epocas fase refinamiento
    'validation_split': 0.2,
    'learning_rate': 1e-3,
    'fine_tune_lr': 1e-5
}

#aqui organizamos las carpetas.
def organize_datasets_automatically():
    print("\n--- Etapa 1: organizando los data sets---")
    try:
        drive.mount('/content/drive')
        base_path = '/content/drive/MyDrive/'
    except:
        base_path = './'

    # estas son las rutas originales en mi drive.
    celeba_original = os.path.join(base_path, 'CelebA')
    my_faces_original = os.path.join(base_path, 'mis_fotos')

    # Rutas de destino
    datasets = {
        'celeba_train': os.path.join(base_path, 'celeba_small/train/data'),
        'my_faces_train': os.path.join(base_path, 'my_faces_small/train/yo'),
        'others_train': os.path.join(base_path, 'my_faces_small/train/otros') #a esto le llaman clase negativa
    }

    # si no existen las carpetas las creamos 
    for folder in datasets.values():
        os.makedirs(folder, exist_ok=True)

    # procesamos con celeba
    all_celeba = list(Path(celeba_original).glob('*.[jJpP]*')) # busca formatos jpg, png, jpeg
    if not all_celeba:
        print(" Error!: No se encontro la carpeta CelebA o esta vacía.")
        return None, base_path

    # copiamos la muestra de celebA para el pre entrenamiento
    if len(os.listdir(datasets['celeba_train'])) < 100:
        print("Copiando imágenes de CelebA...")
        sample = random.sample(all_celeba, min(len(all_celeba), CONFIG['celeba_total_samples']))
        for i, img in enumerate(sample):
            shutil.copy2(img, os.path.join(datasets['celeba_train'], f"img_{i}.jpg"))

    # ahora procesamos las fotos de la persona que quiera usarlo
    my_photos = list(Path(my_faces_original).glob('*.[jJpP]*'))
    if not my_photos:
        print(" Error!: No se encontro la carpeta 'mis_fotos'.")
        return None, base_path

    print(f" Encontradas {len(my_photos)} fotos tuyas.")

    # se limpia la carpeta de destino (por actualizacion)
    for f in os.listdir(datasets['my_faces_train']): os.remove(os.path.join(datasets['my_faces_train'], f))

    #se copian las fotos
    for i, img in enumerate(my_photos):
        shutil.copy2(img, os.path.join(datasets['my_faces_train'], f"yo_{i}.jpg"))

    # se crea la clase "otros" (negativa) usando fotos de CelebA
    # y esto permitira a la red aprender a distinguir
    if len(os.listdir(datasets['others_train'])) < 100:
        print("Creando clase negativa ('otros')...")
        celeba_source = list(Path(datasets['celeba_train']).glob('*.jpg'))[:200] # se usaran 200
        for img in celeba_source:
            shutil.copy2(img, os.path.join(datasets['others_train'], os.path.basename(img)))

    return datasets, base_path

#creamos la arquitectura de nuestro modelo
def create_celeba_pretrain_model():
    # esta sera la red  base para que aprenda caracteristicas faciales 
    model = keras.Sequential([
        layers.Input(shape=(CONFIG['img_height'], CONFIG['img_width'], 3)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid') 
    ])
    return model

def create_final_model(base_model):
    #usamos transfer learning para reusar convolucionales y quitar densas 
    feature_extractor = keras.Sequential(base_model.layers[:-2])
    feature_extractor.trainable = False # congelamos al inicio

    model = keras.Sequential([
        layers.Input(shape=(CONFIG['img_height'], CONFIG['img_width'], 3)),
        feature_extractor,

        # clasificador  con regularizador 
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5), # apagamos  neuronas al azar para evitar memorizacion 
        layers.Dense(1, activation='sigmoid') #el 1 soy yo o la persona a reconocer
        #el 0 son los otrso
    ])
    return model

#entrenamiento
def train_pipeline():
    datasets, base_path = organize_datasets_automatically()
    if not datasets: return None

   #usamos data argumentation para generar fotos mias debido a que tengo solo 40
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,       # rotacion
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,          # zoom
        horizontal_flip=True,
        brightness_range=[0.6, 1.4], # variacion de brillo 
        fill_mode='nearest',
        validation_split=0.2     #esto es para  separar validation auto
    )

  #generadores 
    print("\n--- Configuracion de generadores ---")

    # para celebA
    celeba_gen = train_datagen.flow_from_directory(
        os.path.dirname(datasets['celeba_train']),
        target_size=(CONFIG['img_height'], CONFIG['img_width']),
        batch_size=CONFIG['batch_size'],
        class_mode='binary',
        subset='training'
    )

    # generador de yo vs otros  entrenamiento
    final_train_gen = train_datagen.flow_from_directory(
        os.path.dirname(datasets['my_faces_train']), # usa a my_faces_small/train
        target_size=(CONFIG['img_height'], CONFIG['img_width']),
        batch_size=CONFIG['batch_size'],
        class_mode='binary',
        classes=['otros', 'yo'], #1 para mi 0 para otros
        subset='training',
        shuffle=True
    )

    # generador final para validation sin usar argumentation agresivo
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    final_val_gen = val_datagen.flow_from_directory(
        os.path.dirname(datasets['my_faces_train']),
        target_size=(CONFIG['img_height'], CONFIG['img_width']),
        batch_size=CONFIG['batch_size'],
        class_mode='binary',
        classes=['otros', 'yo'],
        subset='validation'
    )

  #calculo de pesos 
    #forzamos a la red a centrarse en las fotos de la persona 
    count_otros = len(os.listdir(datasets['others_train']))
    count_yo = len(os.listdir(datasets['my_faces_train']))
    total = count_otros + count_yo

    # usamos esta formula para calcularlos Peso = (1 / frecuencia) * (total / 2)
    weight_0 = (1 / count_otros) * (total / 2.0)
    weight_1 = (1 / count_yo) * (total / 2.0)
    class_weights = {0: weight_0, 1: weight_1}

    print(f"\n Pesos calculados : Otros: {weight_0:.2f} | Yo: {weight_1:.2f}")

    # Fase 1
    print("\n FASE 1: aprendiendo características faciales (CelebA)")
    celeba_model = create_celeba_pretrain_model()
    celeba_model.compile(optimizer='adam', loss='binary_crossentropy')
    celeba_model.fit(celeba_gen, epochs=CONFIG['pretrain_epochs'], verbose=1)

    # fase 2 trasnfer learning congelado 
    print("\nFASE 2: Entrenando clasificador (capas base congeladas)")
    final_model = create_final_model(celeba_model)
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history1 = final_model.fit(
        final_train_gen,
        validation_data=final_val_gen,
        epochs=CONFIG['fast_epochs'],
        class_weight=class_weights, #aplicamos los pesos 
        verbose=1
    )

    # Fase 3 fine tuning
    print("\n FASE 3: refinamiento con (Fine-Tuning)")
    feature_extractor = final_model.layers[0]
    feature_extractor.trainable = True # descongelamos 

    # recompilamos con lr bajo
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['fine_tune_lr']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history2 = final_model.fit(
        final_train_gen,
        validation_data=final_val_gen,
        epochs=CONFIG['fine_tune_epochs'],
        class_weight=class_weights, #aplicamos pesos 
        verbose=1
    )

    #guardamos el modelo
    save_path = os.path.join(base_path, 'modelo_desbloqueo_final.h5')
    final_model.save(save_path)
    print(f"\n¡Finalizo en el entranamiento! Modelo guardado en: {save_path}")

    return final_model
#Probamos si funciono
def probar_foto(model_path='/content/drive/MyDrive/modelo_desbloqueo_final.h5'):
    if not os.path.exists(model_path):
        print("Primero debes entrenar el modelo.")
        return

    model = keras.models.load_model(model_path)
    print("\n Sube una foto para entrenar:")
    uploaded = files.upload()

    for fn in uploaded.keys():
        # preprocesamos 
        img = load_img(fn, target_size=(128, 128))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # predecimos 
        score = model.predict(x)[0][0]

        # Mostrar
        plt.figure(figsize=(4,4))
        plt.imshow(load_img(fn))
        plt.axis('off')

        # 
        if score > 0.4:
            plt.title(f"¡ERES TÚ!  (Conf: {score:.1%})", color='green', fontweight='bold')
        else:
            plt.title(f"NO ERES TÚ  (Conf: {score:.1%})", color='red', fontweight='bold')
        plt.show()

#ejecucion esencial
if __name__ == "__main__":
    # entrenamos 
    modelo = train_pipeline()

    # 2. probamos si el entrenamiento termino
    if modelo:
        probar_foto()
