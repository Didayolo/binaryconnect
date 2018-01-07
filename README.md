## Binarisation des poids de réseau neuronnaux durant l'entraînement et applications à la classification d'images

Etude de l'article : "BinaryConnect: Training Deep Neural Networks with binary weights during propagations" par Matthieu Courbariaux, Yoshua Bengio and Jean-Pierre David.

Le principe général est de binairiser (-1 ou +1) les poids d'un réseau neuronal durant les phases de forward et de backward propagation afin d'effectuer moins de calculs. Nous testons ensuite notre modèle avec différentes images, différentes transformations.

[Le Drive contenant les images de test](https://drive.google.com/drive/folders/1WFG3A7NDteLy66UbhyV_aIvolDCU6eB9)

#### Mise en place

Faire `pip install -r requirements.txt`.

Pour faire tourner le programme, il faut d'abord télécharger les images depuis le drive. Ce sont les images de base récupérées sur Google Images. Elles doivent être dans un dossier `Images/Originales`, et la structure avec chaque dossier représentant une classe doit être respectée.

Il faut ensuite lancer `python3 process_images.py`. Si le programme est exécuté tel quel, il va créer automatiquement tous les dossiers (un par transformation, par ex "Blurred" pour les images floutées, "LowContrast", etc.) contenant les nouvelles images. Chacun de ces dossiers contient 10 sous dossiers correspondant aux 10 classes de cifar + 4 autres dossiers avec des images contenants plusieurs objets de cifar ou des images avec des objets qui ne sont pas dans la base de données. 

L'entraînement et les tests des réseaux neuronaux sont dans `keras_cifar10.ipynb`.
