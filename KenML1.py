import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import RandomizedPCA
from itertools import cycle
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import confusion_matrix
from matplotlib import cm
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.decomposition import PCA
from numpy import corrcoef, sum, log, arange
from numpy.random import rand
from pylab import pcolor, show, colorbar, xticks, yticks
#
#Initilialize
# Read the set and preprocess
class ken:
	#pl.rcParams['figure.figsize']= 10,7.5
	pl.rcParams['axes.grid']=True
	#pl.gray()
	pl.close()
#
	def __init__(self,n_sample):
     		self.n_sample=n_sample # set number of parameters
       		self.kfile=pd.read_csv('De-ID-24_061713_newcols.txt').ix[1:self.n_sample,:] # read file
       		self.y=self.kfile.suia # suidice bit
		self.dict={}
		self.dict={i:self.kfile.columns[i] for i in range(0,len(self.kfile.columns))} # build a dictionary easier to build include list
#
		self.remove=[0,1,5,6,7,8,9,10,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,37,38,39,40,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,64,72,73,74,75,76,77,78,79,80,81,82,83,15,16,17,92,98,103,104,105,106]
		# remove the not necessary features
		for r in self.remove:
			#print(r)
			#print(self.dict[r])
			self.kfile.drop([self.dict[r]],inplace=True,axis=1)
		#######
        # Plot ROC curve.
	def make_ROC(self,fpr,tpr,roc_auc,title,fname):
		# Plot ROC curve
		fig,ax = subplots()
		pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' %( roc_auc))
		pl.plot([0, 1], [0, 1], 'k--')
		pl.xlim([0.0, 1.0])
		pl.ylim([0.0, 1.0]); ax.set_xlabel('False Positive Rate')
		pl.title(title)
		pl.legend(loc="lower right");ax.set_ylabel('True Positive Rate')
		pl.savefig(fname)
		pl.show()
        #Plot confusion matrix
	def PlotConfMat(self,cm):
		""""Plot the confusion matrix"""
		normal_cm=(1.0/cm)*np.sum(cm)
		fig,ax=pl.subplots()
		ax.matshow(normal_cm)
		cmap=ax.matshow(normal_cm,interpolation='nearest',cmap=pl.cm.coolwarm)
		cbar=fig.colorbar(cmap,ticks=[-7,0],orientation='vertical')
		cbar.ax.set_yticklabels(['Low','High'])
		pl.title('Confusion Matrix')	
		pl.ylabel('True Label')
		pl.xlabel('Predicted Label')
		pl.show()
	
	# Feature importance using random forest
	def GetRFImportance(self,X,y):
		""" This method will take fit a random forest and plot a barplot of the impotance of the 
			features"""
		forest=ExtraTreesClassifier(n_estimators=100,random_state=0)
		forest.fit(X,y)
		X_new=forest.fit(X,y).transform(X)
		importance=forest.feature_importances_
		indx=np.argsort(importance)[::-1] #sort in decreasing order
		feature_std=np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
		feature_mean=np.mean([tree.feature_importances_ for tree in forest.estimators_],axis=0)
		summary_table=zip(feature_mean[indx],feature_std[indx],X.columns[indx])
		N=X.shape[1]
		width=0.65
		#labels=[feature_dict[db_term] for db_term in list(X.columns[indx])]
		#
		fig, ax =subplots()
		
		ax.set_ylabel('scores from RF fitting')
		ax.set_title('Features by Importance')
		ax.tick_params(axis='x',labelsize=10)
		rect=pl.bar(range(N),feature_mean[indx],width,color='r',yerr=feature_std[indx])	
 		pl.xticks(np.arange(len(X.columns))+0.32,list(X.columns[indx]),rotation=89)
		pl.tight_layout()
		pl.savefig('Importance2.pdf')
		pl.show()
		return indx
	def Best_RF_Feature(self,X,y):
		""" This method will return the best features the software thinks is the best"""
		rf=ExtraTreesClassifier(n_estimators=500,max_features=X.shape[1],random_state=0)
		#
		rf.fit(X,y)
		imp=np.argsort(rf.feature_importances_)[::-1]
		X_new=rf.fit(X,y).transform(X) # return the best number of features
		
		rf=ExtraTreesClassifier(n_estimators=500,max_features=X_new.shape[1],random_state=0)
		#
		X_train, X_test,y_train, y_test=train_test_split(X_new,y,test_size=0.33,random_state=0)
		y_pred=rf.fit(X_train,y_train).predict(X_test)
		cm=confusion_matrix(y_pred,y_test)
		score=rf.score(X_test,y_test)
		return cm,score,X_new,imp

	def RandomForest_Best_ROC(self,X,y):
		""" This method will fit a random forest for the best fit set of features tree
			and draw the ROC curve
			"""
		rf=ExtraTreesClassifier(n_estimators=100,random_state=123)
		
		X_optimal=rf.fit(X,y).transform(X) # Find optimal set.
		rf=ExtraTreesClassifier(n_estimators=100,max_features=X_optimal.shape[1],random_state=123)
		X_train, X_test,y_train, y_test=train_test_split(X_optimal,y,test_size=0.33,random_state=0) # generate training and testing sets

		rf.fit(X_train,y_train)
		cv_score=cross_val_score(rf,X_train,y_train,scoring='roc_auc',cv=5)
		probas=rf.fit(X_train,y_train).predict_proba(X_test)
		y_pred=rf.fit(X_train,y_train).predict(X_test)
		#
		cm=confusion_matrix(y_test,y_pred)
		#
		fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
		roc_auc = auc(fpr, tpr)
		return cv_score, fpr,tpr,roc_auc,cm
        

	def FS_chi(self,X,y,prcnt):
	#
		X=np.array(X)
		y=np.array(y)
				#
		fs=SelectPercentile(chi2,percentile=prcnt)
		#X_chi=fs.fit_transform(X,y)
		X_chi=fs.fit(X,y).transform(X)
		rf=ExtraTreesClassifier(n_estimators=500,max_features=X_chi.shape[1],random_state=0)
		cv_score=cross_val_score(rf,X,y,scoring='roc_auc',cv=5)
		#
		X_train, X_test,y_train, y_test=train_test_split(X_chi,y,test_size=0.33,random_state=0) # get the training and testing sets
		y_pred=rf.fit(X_train,y_train).predict(X_test)
		cm=confusion_matrix(y_pred,y_test)
		return X_chi,cv_score,fs.get_support(),cm
	
	def FS_KBest(self,X,y):
		X=np.array(X)
		y=np.array(y)
                #
		fs=SelectKBest(chi2,k=20)
		X_Kbest=fs.fit_transform(X,y)
		rf=ExtraTreesClassifier(n_estimators=500,max_features=X_Kbest.shape[1],random_state=0)
		cv_score=cross_val_score(rf,X,y,scoring='roc_auc',cv=5)
		#
		X_train, X_test,y_train, y_test=train_test_split(X_chi,y,test_size=0.33,random_state=0) # get the training and testing sets
		y_pred=rf.fit(X_train,y_train).predict(X_test)
		cm=confusion_matrix(y_pred,y_test)
		return X,cv_score,fs.get_support(),cm
	
#	
	def Iter_RF_Importance(self,X,y):
		""" This method will iterate through the set of Important variables and fit a model with the necessary features
		 it return cv_matrix,cv_new,cm, X_new
			cv_matrix  matrix of scores 
			cv_new  tA new transfored matrix of data with the best features from scikit learn
 			X_new= the transforemd X matrix with the new features.
		"""
		n_iter=10
		max_feature=30
		#
		# Compute the importnce 
		forest=ExtraTreesClassifier(n_estimators=100,random_state=0)
		forest.fit(X,y)
		importance=forest.feature_importances_
		indx=np.argsort(importance)[::-1] #sort in decreasing order
		forest_new=ExtraTreesClassifier(n_estimators=100,random_state=0)
		#
		score=np.zeros((X.shape[0],2))
		cv=ShuffleSplit(X.shape[0],n_iter=n_iter,test_size=0.3,train_size=0.7,random_state=0)# generate indices
		#
		cv_matrix=np.zeros((n_iter,max_feature))       
		for features in np.arange(1,max_feature):
			forest_new=ExtraTreesClassifier(n_estimators=100,max_features=features,random_state=0)
			cv_score=cross_val_score(forest,X.ix[:,indx[0:features]],y,cv=cv,scoring='roc_auc')
			cv_matrix[:,features]=cv_score
		#print(np.mean(cv_matrix,axis=0))
		X_new=forest.fit(X,y).transform(X) # Get the best set.
		#
		trees=ExtraTreesClassifier(n_estimators=100,max_features=X_new.shape[1],random_state=0)
		#
		X_train, X_test,y_train, y_test=train_test_split(X_new,y,test_size=0.33,random_state=0)
		trees.fit(X_train,y_train)
		y_pred=trees.fit(X_train,y_train).predict(X_test)
		cm=confusion_matrix(y_pred,y_test)
		cv_new=cross_val_score(trees,X_new,y,cv=cv,scoring='roc_auc')
		#
		return cv_matrix,cv_new,cm, X_new

		
		
	def Iter_RF_Importance_f(self,X,y):
		""" This method will iterate through the set of Important variables and fit a model with the necessary features
		 it return cv_matrix,cv_new,cm, X_new
			cv_matrix  matrix of scores of accuracy
			cv_new  tA new transfored matrix of data with the best features from scikit learn
 			X_new= the transforemd X matrix with the new features.
		"""
		n_iter=10
		max_feature=20
		#
		# Compute the importnce 
		forest=ExtraTreesClassifier(n_estimators=100,random_state=0)
		forest.fit(X,y)
		importance=forest.feature_importances_
		indx=np.argsort(importance)[::-1] #sort in decreasing order
		forest_new=ExtraTreesClassifier(n_estimators=100,random_state=0)
		#
		score=np.zeros((X.shape[0],2))
		cv=ShuffleSplit(X.shape[0],n_iter=n_iter,test_size=0.3,train_size=0.7,random_state=0)# generate indices
		#
		cv_matrix=np.zeros((n_iter,max_feature))       
		for features in np.arange(1,max_feature):
			forest_new=ExtraTreesClassifier(n_estimators=100,max_features=features,random_state=0)
			cv_score=cross_val_score(forest,X.ix[:,indx[0:features]],y,cv=cv,scoring='roc_auc')
			cv_matrix[:,features]=cv_score
		#print(np.mean(cv_matrix,axis=0))
		X_new=forest.fit(X,y).transform(X) # Get the best set.
		#
		trees=ExtraTreesClassifier(n_estimators=100,max_features=X_new.shape[1],random_state=0)
		#
		X_train, X_test,y_train, y_test=train_test_split(X_new,y,test_size=0.33,random_state=0)
		trees.fit(X_train,y_train)
		y_pred=trees.fit(X_train,y_train).predict(X_test)
		cm=confusion_matrix(y_pred,y_test)
		cv_new=cross_val_score(trees,X_new,y,cv=cv,scoring='f1')
		#
		return cv_matrix,cv_new,cm, X_new
		
		
	def Plot_Errbar(self,cv_matrix,title,xlab,ylab,fname):
		""" This method recieves a cv matrix object and returns its plot with error bars"""
		#
		fname=fname+'.pdf'
		std=np.std(cv_matrix[:,1:],axis=0)
		mean=np.mean(cv_matrix[:,1:],axis=0)
		#
		pl.close('All')
		fig, ax =subplots()
		ax.set_title(title)
		ax.set_xlabel(xlab)
		ax.set_ylabel(ylab)
		#
		pl.errorbar(np.arange(1,1+mean.shape[0]),mean,yerr=std)
		pl.savefig(fname)		
	def Iter_PCA(self,X,y,max_features):
		"""This methods will try to model the data based on PCA, adding more features as we go alongand compute the  ROC for them"""
		n_iter=10
		pca=PCA(whiten=True)
		X_new=pca.fit_transform(X) # transform the data to the new corrdinates
		cv=ShuffleSplit(X.shape[0],n_iter=n_iter,test_size=0.3,train_size=0.7,random_state=0)# generate indices
		cv_matrix=np.zeros((n_iter,max_features))	
		for i,features in enumerate(np.arange(1,max_features+1)):		
			print ([i,features])
			forest=ExtraTreesClassifier(n_estimators=100,max_features=features,random_state=0)
			cv_score=cross_val_score(forest,X_new[:,:features],y,cv=cv,scoring='roc_auc')
			cv_matrix[:,i]=cv_score	
		return cv_matrix
		#k.Plot_Errbar(cv_matrix,'Iteration Principal Components','PCs','ROC AUC','IterPCA')
	def Iter_PCA_ACC(self,X,y,max_features):
		"""This methods will try to model the data based on PCA, adding more features as we go alongand compute the  ROC for them"""
		n_iter=10
		pca=PCA(whiten=True)
		X_new=pca.fit_transform(X) # transform the data to the new corrdinates
		cv=ShuffleSplit(X.shape[0],n_iter=n_iter,test_size=0.3,train_size=0.7,random_state=0)# generate indices
		cv_matrix=np.zeros((n_iter,max_features))	
		for i,features in enumerate(np.arange(1,max_features+1)):		
			print ([i,features])
			forest=ExtraTreesClassifier(n_estimators=100,max_features=features,random_state=0)
			cv_score=cross_val_score(forest,X_new[:,:features],y,cv=cv,scoring='precision')
			cv_matrix[:,i]=cv_score	
		return cv_matrix	
		
		
			
	def PCA_var(self,X,y,N):
		""" This methods does a PCA decomposition and returns the variance for each component"""
		pca=PCA(whiten=True)
		pca.fit(X)
		width=0.51
		fig=pl.figure(1)
		ax1=pl.subplot(211)
		pl.subplots_adjust(hspace=0.5)
		ax1.set_title('Variance explaind by PCA')
		ax.tick_params(axis='x',labelsize=10)
		ax1.set_xlim([0,N])
		ax1.set_xlabel('principal componens')
		ax1.set_ylabel('percent of variability explained')
		ax1.set_xticks(np.arange(N)+width/2)
		ax1.set_xticklabels(tuple(str(l) for l in range(N)))
		#ax.bar(arange(pca.explained_variance_ratio_[0]),pca.explained_variance_ratio_,0.65,color='r')
		ax1.bar(np.arange(N),pca.explained_variance_ratio_[:N],width,color='r')
		#
		ax2=pl.subplot(212)
		ax2.set_title('Variance explaind by PCA')
		ax2.tick_params(axis='x',labelsize=10)
		ax2.set_xlim([0,N])
		ax2.set_xlabel('principal componens')
		ax2.set_ylabel('percent of variability explained')
		#pl.xticks(np.arange(len(X.columns))+0.32,list(X.columns[indx]),rotation=89)
		ax2.set_xticks(np.arange(N)+width/2)
		ax2.set_xticklabels(tuple(str(l) for l in range(N)))
		#ax.bar(arange(pca.explained_variance_ratio_[0]),pca.explained_variance_ratio_,0.65,color='r')
		ax2.bar(np.arange(N),np.cumsum(pca.explained_variance_ratio_[:N]),width,color='r')
		pl.savefig('VarPCA.pdf')
		
	def PCA2d(self,X,y,xlab,ylab):
		""" Project the data set onto a lower space of 2D"""
		col=['r','y']
		c=[col[i] for i in y]
		c_max=5
		pca=PCA(n_components=c_max,whiten=True)
		X_new=pca.fit_transform(X)
			
		fig,ax=pl.subplots()
		ax.set_title('Projecting to 2D using PCA')
		ax.set_ylabel(ylab)
		ax.set_xlabel(xlab)
		pl.scatter(X_new[:,1],X_new[:,2],c=c)
		pl.savefig('PCACorr2.pdf')
		return X_new
		
	def PCA_Corr(self,X,y,n_samples,n_features):
		""" Plot a heta map  of the correlation matrix
		"""
		n_samples=1000
		n_features=20
		pca=PCA(whiten=True)
		X=pca.fit_transform(X)
		yy=np.array(y[:n_samples]).reshape(n_samples,1)
		xx=np.array(X[:n_samples,:n_features])
		x=np.append(yy,xx,axis=1)[:,:n_features]
		#
		R =np.corrcoef(x.T)
		#
		end=x.shape[1]
		fig,ax=pl.subplots()
		#pl.matshow(R)
		pl.pcolor(R)
		colorbar()
		pl.xticks(arange(0.5,end),range(1,x.shape[1]+1))
		pl.yticks(arange(0.5,end),range(1,x.shape[1]+1))
		ax.set_title('Corrolelorgam of suicide and  PCs')
		pl.savefig('CorrFig.pdf')
		show()
		
######################################
k=ken(15000)
X=k.kfile
y=k.y

