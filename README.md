process_images : lancer avec python3. permet d'appliquer les transformations aux images situées dans le dossier Images. Il faut qu'il contienne au moins le dossier nommé "Original" avec les images de base récupérées sur Google Images. Si le programme est exécuté tel quel, il va créer automatiquement touts les dossiers (un par transformation, par ex "Blurred" pour les images floutées, "LowContrast", etc.) contenant les nouvelles images. Chacun de ces dossiers contient 10 sous dossiers correspondant aux 10 classes de cifar + 4 autres dossiers avec des images contenants plusieurs objets de cifar ou des images avec des objets qui ne sont pas dans la base de données. 

rename.sh : Permet de renommer toutes les images d'un dossier. Prend en argument un nom d'image, par exemple "cat", et va renommer toutes les images en "cat1", "cat2", etc.

keras-cifar10.ipynb : 


xxx.xx : le modèle sauvegardé. # Binary Connect

## Binarisation des poids de réseau neuronnaux durant l'entraînement et applications à la classification d'images

Etude de l'article tanani ...

Mise en place ...
Explications des fichier, requirements...

[Lien vers le Drive](https://drive.google.com/drive/folders/1WFG3A7NDteLy66UbhyV_aIvolDCU6eB9)

Faire `pip install requirements.txt`.

Pour faire tourner le programme, il faut d'abord récupérer les images sur le drive. Ce sont les images de base récupérées sur Google Images. Elles doivent être dans un dossier `Images/Originales`, et la structure avec chaque dossier représentant une classe doit être respectée.

Il faut ensuite lancer `python3 process_images.py`. Si le programme est exécuté tel quel, il va créer automatiquement tous les dossiers (un par transformation, par ex "Blurred" pour les images floutées, "LowContrast", etc.) contenant les nouvelles images. Chacun de ces dossiers contient 10 sous dossiers correspondant aux 10 classes de cifar + 4 autres dossiers avec des images contenants plusieurs objets de cifar ou des images avec des objets qui ne sont pas dans la base de données. 

L'entraintement et les tests des réseaux neuronaux sont dans `keras_cifar10.ipynb`.
