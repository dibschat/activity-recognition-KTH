import os
import sys
import math
import pandas as pd
import numpy as np 
import scipy
import cv2
import imutils
from skimage import exposure, feature
from shapely.geometry import Point, Polygon
from classifier import SVM, SVM_grid, Ensemble
from motion_2x2 import dense_flow as motion_2x2
from motion_3x3 import dense_flow as motion_3x3
from motion_4x4 import dense_flow as motion_4x4
from contextual_2x2 import dense_flow as contextual_2x2
from contextual_3x3 import dense_flow as contextual_3x3
from contextual_4x4 import dense_flow as contextual_4x4
from orientation_features import HOG_central, HOG_context

if __name__=="__main__":
	process = sys.argv[1]

	# set the root directory where the dataset is present
	rootDir = '/media/dibyadip/DC/Project Work/SKS/activity_recognition/KTH'

	if(process=="process"):# for processing videos and creating features (individual/ensemble)
		Flag = False
		feature_vector = np.array([])
		count = -1

		for dirName, subdirList, fileList in os.walk(rootDir):
			str = ""
			count = count+1 

			is_first = True
			for fname in fileList:				
				str = dirName+'/'+fname
				a = motion_2x2(str)
				b = motion_3x3(str)
				c = motion_4x4(str)
				d = HOG_central(str)
				# an added context feature is also processed for motion to get consistency between the valid videos processed
				e, f_n = contextual_2x2(str)

				#a, f_na = contextual_2x2(str)
				#b, f_nb = contextual_3x3(str)
				#c, f_nc = contextual_4x4(str)
				#d = HOG_context(str)

				#b = np.concatenate((a,b), axis=0)
				#b = np.concatenate((a,b,c), axis=0)
				b = np.concatenate((a,b,c,d), axis=0)			

				count_final = np.array([count],dtype='int')

				b = np.concatenate((b,count_final),axis=0)	
	  
				if Flag==False and not feature_vector:
					feature_vector = np.array([b])
					Flag=True
					
				else:
					if(b.shape[0]==feature_vector.shape[1] and e.shape[0]==280):
						feature_vector = np.vstack((feature_vector,b))
					#if(b.shape[0]==feature_vector.shape[1]):
						#feature_vector = np.vstack((feature_vector,b))
					
				print(feature_vector.shape)
				is_first=False
		print(feature_vector.shape)

		# save the create feature in the bin directory
		np.save("./bin/feature_vector_KTH_motion+HOG.npy", feature_vector)

	elif(process=="concat_features"):# for concatenating individual features (+)
		feature_vector1 = np.load('./data/feature_vector_KTH_context_trio_central.npy')
		feature_vector2 = np.load('./data/feature_vector_KTH_HOG_context_central.npy')

		print(feature_vector1.shape)
		print(feature_vector2.shape)

		feature_vector1 = np.delete(feature_vector1, -1, 1)
		feature_vector = np.hstack((feature_vector1, feature_vector2))

		# save the create feature in the bin directory
		np.save("./data/feature_vector_KTH_motion+contextual.npy", feature_vector)

	elif(process=="train_ensemble"):# for classifier ensembling (*)
		feature_vector1 = np.load('./bin/feature_vector_KTH_motion+HOG.npy')
		feature_vector2 = np.load('./bin/feature_vector_KTH_contextual.npy')
		Ensemble(feature_vector1, feature_vector2)
		exit()

	elif(process=="train"):# for training individual features
		feature_vector = np.load('./bin/feature_vector_KTH_motion+HOG.npy')

		# for training and classification
		SVM(feature_vector)

		# for grid search
		#SVM_grid(feature_vector)
