# SIMPLON DEV IA | Brief 12


## Projet de détection de tumeurs cérébrales pour la startup NeuroGuard

### Contexte

En tant que développeur IA, il nous a été demandé de développer une solution se basant sur l'IA pour la détection de tumeurs cérébrales.

Consignes :

- Effectuer des prétraitements d'images : appliquer des techniques d'OpenCV pour normaliser, redimensionner et améliorer la qualité des images médicales. Les données seront également augmentées pour améliorer les prédictions.
​
- Utiliser VGG-16 : vous allez d'abord explorer l'architecture de VGG-16 puis l'adapter à la détection de tumeurs cérébrales.

### Structure du projet

```bash
project/
│
├── functions/    # Python scripts for each function used in notebook
│
├── img/
│   ├── raw/    # Raw images directory
│   │   ├── yes/
│   │   └── no/
│   │
│   └── splits/    # Splitted data directory
│       ├── test/
│       │   ├── yes/    # Images with tumor
│       │   └── no/    # Images without any tumor
│       │
│       ├── train/
│       │   ├── yes/
│       │   └── no/
│       │
│       └── val/
│           ├── yes/
│           └── no/
│
├── .gitignore
├── main.ipynb    # Main notebook
└── README.md
```

# Utilisation de OpenCV

Nous utilisons OpenCV pour lire les images, les encoder et appliquer des modifications dessus.

# Data-split et affichage du split

Nous commençons par "split" la data : grâce aux modules "os" et "train_test_split", nous divisons la database en en 3 dossiers : "train", "test" et "val".

Nous créons un dataframe pour afficher la listes de nos images avec les informations sur la présence de tumeur et le dossier dans lequel elles se situent.

Nous chargons le modèles avec les X et y de chaque base de données: "train","test" et "val": X, étant l'image et y, étant l'information si il y a présence de tumeur en "0" (abscence de tumeur) ou "1" (tumeur).

![Dataframe_total](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/be4291dd-e789-4a96-ada4-afc4d24daedb)

# Normalisation des images

Nous appliquons une normalisation sur les images pour que elles soient toutes centrées et de la même dimension de sorte à ce que les dimensions du crâne soit les mêmes sur toutes les photos : condition nécéssaire pour les "loader" dans notre modèle "VGG16". Un filtre OpenCV "GaussianBlur" est appliqué pour réduire le bruit sur chaque image.

![No Filter_plot](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/3fe166e1-5e42-4d56-b459-38c200298846)
![Sharpenning_plot](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/f5552b31-3550-4510-8471-d22cd9891697)
![Sharpenning_and_Sobel_y_plot](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/5a54b7f7-60dc-422f-89c7-0ce682f7ab0a)
![Sobel_plot](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/869d67ca-095c-422a-a730-fd52e3a0aa8f)
![Sobel_x_plot](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/83dfca15-9889-4009-9ee2-d093d094d84d)
![Sobel_y_plot](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/7bfcb0ae-1f32-4f61-b261-244fa31d7a93)


On utlise la fonction "normalize":

```py
normalize_images(X, target_size, apply_sharpening=False, apply_sobel=False, apply_sobel_x=False, apply_sobel_y=False, sobel_k_size=5)
```

La fonction a pour options :
apply_sharpening: appliquer un filtre de netteté.
apply_sobel : appliquer un filtre "Sobel"(SobelX + SobelY).
apply_sobel_x : appliquer un filtre "Sobel X"
apply_sobel_y : appliquer un filtre "Sobel Y"
sobel_k_size : changer la taille du "kernel".

Le Sobel est un filtre de OpenCV, c'est un opérateur utilisé en traitement d'image pour la détection de contours. 


# Training du modèle "VGG16"

Nous loadons le modèle "VGG16" avec les images normalisées : 

```py
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
```

La base "train" sert à entrainer le modèle.
La base "val" sert à effectuer des validations du modèle de sorte à générer une courbe "loss".
La base "test" sert à effectuer des tests pour obtenir un score "accuracy" ou "Précision".

La précision est le ratio du nombre de vrais positifs par rapport au nombre total de prédictions positives. Par exemple, si le modèle a détecté 100 tumeurs, et qu’il y en a 90, la précision est de 90 pour cent.

Precision = (True Positive)/(True Positive + False Positive)


## Tests et résultats


Hyperparamètres : <br>
- Split 1 : "60% train / 20% test / 20% val" <br>
- Split 2 : "80% train / 10% test / 10% val" <br>


Options : <br>
- apply_sharpening : "S" <br>
- apply_sobel : "So" <br>
- apply_sobel_x : "SoX" <br>
- apply_sobel_y : "Soy" <br>
- sobel_k_size : "k" <br>

Nous avons initialisé des tests en changeant les paramètres de la taille de la base "train" et du filtre "apply_sharpening":

(test_1) VGG16/ Split 1 / Epoch = 10 (): loss: 7.6316 - accuracy : 0.7451 
![VGG16_ split_1_epoch_10_accuracy](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/f1840f33-ff90-4797-8fa2-a74d58a504e5)
![VGG16_ split_1_epoch_10_loss](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/3c9f42f5-59a8-459e-ac8a-ce3461656861)

(test_2) VGG16/ Split 1 / Epoch = 10 (): loss: 14.3125 - accuracy: 0.6579
VGG16/ Split 1 / Epoch = 10 (S) : loss: 11.2359 - accuracy : 0.7255 


(test_1) VGG16/ Split 2 / Epoch = 10 (): loss: 5.5376 - accuracy: 0.8026
![VGG16_ split_2_epoch_10_accuracy](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/41587037-133b-402a-ba96-153a9035a55c)
![VGG16_ split_2_epoch_10_loss](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/a67ca9ce-007d-49f2-853d-52d648a25834)

(test_2) VGG16/ Split 2 / Epoch = 10 ():  loss: 9.5727 - accuracy: 0.7105 <br>
VGG16/ Split 2 / Epoch = 10 (S) : loss: 6.4138 - accuracy : 0.6842

Puis nous avons initialisé des tests en changeant les paramètres de "epoch" du modèle "VGG16":

VGG16 / Split 1 / Epoch = 10 (): accuracy : 0,69
VGG16 / Split 1 / Epoch = 20 (): accuracy : 0,73

(test_1) VGG16 / Split 1 / Epoch = 30 (): accuracy : 0,84 
![VGG16_ split_2_epoch_30_loss_2](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/63ead939-f4f0-43d2-95e3-69263b10208a)

(test_2) VGG16 / Split 1 / Epoch = 30 (): loss: 6.8281 - accuracy: 0.8158
![VGG16_ split_2_epoch_30_loss](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/b387fdfb-bf21-448e-a2e0-4766964d8629)
![VGG16_ split_2_epoch_30_accuracy](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/b07f3c52-12d7-44b6-9a0e-753384ba3650)

(test_3) VGG16 / Split 1 / Epoch = 30 (): loss: 12.2155 - accuracy: 0.7763
![VGG16_ split_2_epoch_30_loss_2](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/bccb3a61-4555-400b-80f9-dd6516a83d16)
![VGG16_ split_2_epoch_30_accuracy_2](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/0655b4a7-97d0-4ceb-85fa-08f9209d7ecf)



Nous avons remarqué un pic du "Trainning Accuracy" pour epoch=24 sur le graphique du "(test_2) VGG16 / Split 1 / Epoch = 30 ()", nous essayé un test avec ce paramètre :

VGG16 / Split 1 / Epoch = 24 (): loss: 7.2410 - accuracy: 0.7895
![VGG16_ split_2_epoch_24_loss](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/45265ad8-34df-4858-955a-abe3f1e678c2)
![VGG16_ split_2_epoch_24_accuracy](https://github.com/Sandalcho7/simplon_brief12_neuroguard/assets/154426850/5c8d9b53-0cd0-4827-81bc-44db34691286)


## Conclusion

On peut observer que les paramètres de "train_size" et de "Epoch" ont une relation de proportionnalité avec l'indice "accuracy", plus ils augmentent,
plus l'indice augmente donc plus notre modèle est meilleur. On peut observer que le filtre "apply_sharpening" a tendance a détériorer l'indice "accuracy".


## Commentaire

Nous allons orienter nos tests dans l'ajustement des paramètres de "train_size" et de "Epoch" et de les combiner pour obtenir des tests avec de meilleurs indices "accuracy".

Les autres filtres OpenCV tel que "Sobel", "Sobel_X" et "Sobel_Y" ne peuvent pas être implantés directement malgrès que les images s'affichent correctement, ces filtres change la structure("shape") de l'image et donc ne peut pas être interprétés correctement par le modèle "VGG16". Nous travaillont à une adaptation de ces images aux modèles pour lui permettre de s'entrainer avec des images résultants de ces filtres.

Nous avons choisi ces filtres OpenCV car ils sont opérateurs utilisés en traitement d'image pour la détection de contours, donc adaptés à la détection de tumeurs cérébrales.
