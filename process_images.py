import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import util, transform, exposure
import random
import scipy
import os

"""Ajout de flou gaussien"""
def gaussian_blur(img):
	return cv2.GaussianBlur(img,(5,5),20)

"""Ajout de bruit gaussien"""
def add_noise(image):
	return util.random_noise(image, mode='gaussian', seed=None, clip=True)

"""Effectue une rotation aléatoire de 90, 180 ou 270 degrés"""
def rotate_img(img):
	angles = [90, 180, 270]
	angle = angles[random.randint(0,len(angles)-1)]
	return transform.rotate(img, angle, resize=True)

"""Changement de contraste. contrast_type doit prendre les valeurs high, soft ou low """
def change_contrast(img, contrast_type):
	if contrast_type == "high": #Augmentation forte du contraste
		return exposure.rescale_intensity(img, in_range=(40, 170))
	if contrast_type == "soft": #légère augmtnetation du contraste
		return exposure.rescale_intensity(img, in_range=(15, 235))
	return exposure.rescale_intensity(img, out_range=(80, 220))#diminution du contraste

""" Inversion des couleurs """
def invert_colors(img):
	return (255-img)

def print_image(img, blur):
	plt.subplot(121),plt.imshow(img),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(blur),plt.title('Transformed')
	plt.xticks([]), plt.yticks([])
	plt.show()



""" Resize les images en 32x32 et applique les transformations avant si nécessaire"""
def resize_all_images(blur=False, noisy=False, rotate=False, high_contrast_increase=False,  low_contrast_increase=False,contrast_decrease=False,color_inversion=False):
	destination = 'Resized' #par défaut
	source='Original'
	if(blur and noisy):
		destination='NoisyBlur'
		print("Applying noise and blur")
	elif(rotate and noisy):
		destination = 'NoisyRotated'
		print("Applying rotation and noise")
	elif(rotate):
		destination = 'Rotated'
		print("Applying rotation")
	elif(blur): 
		destination = 'Blurred'
		print("Applying blur")
	elif(noisy):
		destination = 'Noisy'
		print("Applying noise")
	elif(high_contrast_increase):
		destination = 'HighContrast'
		print("Applying high contrast increase")
	elif(low_contrast_increase):
		destination = 'LowContrast'
		print("Applying low contrast increase")
	elif(contrast_decrease):
		destination = 'ContrastDecrease'
		print("Applying contrast decrease")
	elif(color_inversion):
		destination = 'ColorsInverted'
		print("Applying colors inversion")

	#On vérifie si le dossier de destination existe ou pas
	if not os.path.exists('Images/' + destination):
		os.makedirs('Images/' + destination)
	
	for dossier, sous_dossiers, fichiers in os.walk("Images/" + source):
		split = dossier.split("/")
		classe = ""
		if (len(split) >=3):
			classe = split[2]
			print("Processing " + classe + "...")
			if not os.path.exists('Images/' + destination + "/" + classe):
				os.makedirs('Images/' + destination + "/" + classe)
		for fichier in fichiers:
			path = dossier + "/"+ fichier
			img = cv2.imread(path)

			##On transforme les images si elles doivent l'etre
			if blur:
				img = gaussian_blur(img)
			if noisy:
				img = add_noise(img)
			if rotate:
				img = rotate_img(img)
			if high_contrast_increase:
				img = change_contrast(img, "high")
			if low_contrast_increase:
				img = change_contrast(img, "soft")
			if contrast_decrease:
				img = change_contrast(img, "low")
			if color_inversion:
				img = invert_colors(img)

			img_resized = scipy.misc.imresize(img, (32, 32))
			#print_image(img, img_resized)
			cv2.imwrite('Images/' + destination + '/' + classe + "/" + destination + "_" + fichier, img_resized)


if __name__ == "__main__":

	######## Tests ##########
	#Blur
	"""
	img = cv2.imread('example.jpg')
	img_blur = gaussian_blur(img)
	#print_image(img, img_blur)
	
	#Noise
	noisy = add_noise(img)
	#print_image(img, noisy)
	noisy_blur = add_noise(img_blur)
	#print_image(img, noisy_blur)
	cv2.imwrite('blurred.png',img_blur)

	#Rotate
	rotated = rotate_img(img)
	#print_image(img, rotated)
	rotate_noise = add_noise(rotated)
	#print_image(img, rotate_noise)
	
	#Contrast
	img = cv2.imread('example.jpg')
	contrast = change_contrast(img, "soft")
	print_image(img, contrast)

	#Changement de couleurs
	colors = invert_colors(img)
	print_image(img, colors)
	"""
	########### Resize #############
	resize_all_images()

	######### Blur + resize ##########
	resize_all_images(blur=True)

	######### Noise + resize ##########
	resize_all_images(noisy=True)
	
	######### Noise + Blur + resize ##########
	resize_all_images(blur=True,noisy=True)
	
	######### Rotate + resize ##########
	resize_all_images(rotate=True)

	######### Noise + resize ##########
	resize_all_images(noisy=True,rotate=True)

	
	######## High contrast increase + resize ##########
	resize_all_images(high_contrast_increase=True)

	######## Low contrast increase + resize ##########
	resize_all_images(low_contrast_increase=True)

	######## contrast decrease + resize ##########
	resize_all_images(contrast_decrease=True)

	######## color inversion + resize ##########
	resize_all_images(color_inversion=True)
	




	
