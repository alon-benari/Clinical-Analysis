import numpy as np
import random as random
from matplotlib import pyplot as plt
import pandas as pd
import scipy.stats
from  lifelines import KaplanMeierFitter 
from  lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
#
# This script will make all the figures for the manuscript
#
w5 = pd.read_csv('WorkDf4.csv',sep=';')
#
plt.close('all')
def plot_age_hist(w5):
    opioid = w5[(w5.chronic==1)]
    non_opioid = w5[~(w5.chronic==1)]
    ax = (opioid.Age).hist(bins=50,alpha=0.8,color='r',label = 'opioid+')
    ax = (non_opioid.Age).hist(ax=ax,bins=50,alpha=0.5,color='b',label = 'opioid-')
    ax.set_title('cohort age distribution')
    ax.set_xlabel('age')
    ax.set_ylabel('Number of patient')
    ax.legend(loc=2)
    plt.savefig('age_hist.pdf')
    plt.show()

def cox_fit(w5):      
    df=pd.DataFrame({'t2rev':(w5.t2rev),'r':(w5.RevLogical),'chronic':(w5.chronic),'dm':(w5.Diabetes),'age':(w5.Age)})#,'wt':(w5.wt)}) # set the df for cox ph fitting
    cf= CoxPHFitter()
    cf.fit(df,duration_col='t2rev',event_col='r')
    cf.print_summary()

    

def plot_hist2d(df):
    plt.close('all')
    chronic = df[df.chronic==1]
    x = np.array(chronic.med_load)
    y = np.array(chronic.TimeBefore)
    #
    bx=14
    by =35
    fig  = plt.figure(figsize(10,8))
    ax1 = fig.add_axes([0.08,0.07,0.58,0.5])
    ax1.grid(True)
    ax1.scatter(log(x/12),y,c='r',s=2,marker='+')
    ax1.set_title('(c)',fontsize=10)
    z,xedges,yedges = np.histogram2d(log(x/12),y,bins=[14,35],normed=1)
    extent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
   
    H = np.flipud(np.rot90(z))
    cp = ax1.contourf(H,35,extent=extent,alpha=0.6)# ,cmap = plt.cm.rainbow)
    ax1.set_xlim([xedges[0],xedges[-1]])
    ax1.set_ylim(yedges[0],yedges[-1])
    ax1.set_xlabel('log(MED/month)',fontsize=10)
    ax1.set_ylabel('cumulative time(month)',fontsize=10)
    cbar = fig.colorbar(cp,ticks = [0,0.045])
    cbar.ax.set_yticklabels(['low', 'high'])
    #
    ax2 = fig.add_axes([0.08,0.655,0.46,0.3])
    ax2.hist(log(x/12),bins=bx,color='r',alpha=0.5,normed=True)
    ax2.set_ylabel('probability',fontsize=10)
    ax2.set_title('(a)',fontsize=10)
    ax2.set_xlabel('log(MED/month)',fontsize=10)
    
    ax2.grid(True)
    ax2.set_xlim([xedges[0],xedges[-1]])
    #
    ax3=fig.add_axes([0.67,0.07,0.3,0.5])
    ax3.hist(y,bins=by,color='r',orientation='horizontal',alpha=0.5,normed=True)
    

    ax3.set_xlabel('probability',fontsize=10)
    ax3.set_title('(b)',fontsize=10)
    ax3.set_ylabel('cumulative time(month)',fontsize=10)
    ax3.grid(True)
    ax3.set_ylim([yedges[0],yedges[-1]])
    #
    ax4 = fig.add_axes([0.67,0.655,0.3,0.3])
    count,bins = np.histogram(chronic.TypeBefore,bins=7,normed=True)
    ax4.bar(np.arange(1,8),count/sum(count),align='center',color='r',alpha=0.5)
    ax4.set_xlabel('Number of types of opioid/patient',fontsize=10)
    ax4.set_ylabel('probability',fontsize=10)
    ax4.set_title('(d)',fontsize=10)
    ax4.grid(True)
    fig.savefig('2d_hist.pdf')
    fig.show()
    
   


def plot_opioid_rev(dataset):
    '''returns the cumulative amounts of knee revision between  opioid naive and long term opioids.'''
    set=pd.DataFrame({'RevLogical':dataset['RevLogical'],
							  'EarlyRevDate':dataset['EarlyRevDate'],
							  'OpioidsBefore':dataset['IfBefore']})
    idx=[(l!='0')  for i,l in enumerate(set['EarlyRevDate'])]
			#
    data=set[idx].set_index(pd.DatetimeIndex(set[idx]['EarlyRevDate']))
    rev_by_date=data.groupby([data.index]).sum()
    rev_by_date['rev_cumsum']=rev_by_date['RevLogical'].cumsum()
		#
    opioids=pd.DataFrame({'naive':rev_by_date[rev_by_date['OpioidsBefore']==0]['RevLogical'].cumsum(),'chronic':rev_by_date[rev_by_date['OpioidsBefore']==1]['RevLogical'].cumsum()})
    plt.close()         
							  
    fig,ax=plt.subplots(nrows=1,ncols=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('cumulative number')
    opioids.plot(ax=ax,color =['r' ,'b'],figsize=(8,6),title='Cumulative TKRs VA HCS',legend =True)
    ax.legend(('opioid+','opioid-'),loc=0)
    plt.tight_layout()
    plt.savefig('opioid_rev.pdf',dpi=400)
    plt.show()
	
def plot_bars(w5):
    '''Shows the proportion in revision between opioid naive and long term opioids'''
    opioid = w5[(w5.chronic==1)]
    non_opioid = w5[~(w5.chronic==1)]
    #
    long_term = [(opioid.RevLogical).sum(),(opioid.shape[0]-(opioid.RevLogical).sum())]
    
    naive = [(non_opioid.RevLogical).sum(),(non_opioid.shape[0]-(non_opioid.RevLogical).sum())]
    m = np.array([long_term,naive])
    M = np.dot(m,np.diag(1.0/sum(m,axis=0)))
    lt = M[0,:]
    nv =M[1,:]
    #
    N=len(nv)
    ind = np.arange(N)
    width=0.25
    fig,ax = plt.subplots()
    bars1 = ax.bar(ind+0.25*width,lt.tolist(),width,color='r',label='opioid+')
    bars0 = ax.bar(ind+1.25*width,nv.tolist(),width,color='b',label='opioid-')
    ax.set_ylabel('Number of patients')
    ax.set_title('proportion of revision among opioid users')
    ax.set_xticks((ind)+1.25*width)
    ax.set_xticklabels(('revision+','revision-'))
    ax.legend(loc=0)
    ax.grid(True)
    plt.show()
    plt.savefig('bar_fig.pdf')


def plot_num_rev(w5):
    '''The figure shows  the number of revision procedures or opioid/non_opioid gourp. '''
    opioid = w5[(w5.chronic==1)]
    non_opioid = w5[~(w5.chronic==1)]
    count1 = [(opioid.NumRev==p).sum() for p in np.arange(1,7,1)]
    count0 = [(non_opioid.NumRev==p).sum() for p in np.arange(1,7,1)]
    N=6
    ind = np.arange(N)+1
    width = 0.25
    fig,ax = plt.subplots()
    bars1 = ax.bar(ind+0.5*width, count1,width, color='r',label = 'opioid+' )
    bars0 = ax.bar(ind+1.5*width, count0,width, color='b',label = 'opioid+' )
    ax.legend(loc=0)
    ax.set_xticks(np.arange(N)+1.37)
    ax.set_xticklabels(np.arange(1,7,1))
    ax.set_xlabel('# of procedures post arthroplasty')
    ax.set_ylabel('count')
    ax.grid(True)
    plt.savefig('num_rev.pdf')
    plt.show()

def plot_KM_estimator(w5):
    
    """"A method to make KM plots  it returns a plot of the two graphs and the logrank test result"""
    opioid = w5[(w5.chronic==1)]
    non_opioid = w5[~(w5.chronic==1)]
    #
    duration0 = non_opioid.t2rev
    duration0= duration0.ix[duration0>0]

    duration1 = opioid.t2rev
    duration1 = duration1.ix[duration1>0]
	#
    C1 = opioid.RevLogical
    C1=C1.ix[duration1.index]
    C0 = non_opioid.RevLogical
    C0 = C0.ix[duration0.index]
    #
    ax1 = plt.subplot(111)
    ax1.set_ylim = ([0.9,1])
    ax1.set_xlim = ([0,365])
    kmf_naive = KaplanMeierFitter()
    ax1 = kmf_naive.fit(duration0,C0,label='opioid-').plot()
    ax1.set_title('knee revision free survival')
    ax1.set_xlabel('days after arthroplasty')
    ax1.set_ylabel('revision free probability')
    kmf_chronic= KaplanMeierFitter()
    ax1 = kmf_chronic.fit(duration1,C1,label='opioid+').plot(ax=ax1,color='r')
    plt.show
    plt.savefig('KM_rev.pdf')
    summary,p_val, test= logrank_test(duration1,duration0,C1,C0,alpha=0.99)
    return summary

def plot_km_mnp(w5):
	days = np.zeros((df.shape[0],1))
	c=pd.to_datetime(['01/01/2013'])
	tka_date = pd.to_datetime(df['TKADate'])
	mnp_date = pd.to_datetime(df['EarlyMnpDate'],coerce = True)
	for i,r in enumerate(mnp_date):
		if pd.isnull(r):
			days[i] = list((c-tka_date.iloc[i]).days)[0]
		else:
			days[i] = (mnp_date.ix[i]- tka_date.ix[i]).days
	
	df.t2mnp = [d[0] for d in days.tolist()]  #time to possible knee manipualtion
                #
        opioid = df[(df.chronic==1)]
        non_opioid = df[~(df.chronic==1)]
        #
        
        duration0 = non_opioid.t2mnp
        duration0= duration0.ix[duration0>0]

        duration1 = opioid.t2mnp
        duration1 = duration1.ix[duration1>0]
	#
        C1 = opioid.ManipLogical
        C1=C1.ix[duration1.index]
        C0 = non_opioid.ManipLogical
        C0 = C0.ix[duration0.index]
        #
        
        ax1 = plt.subplot(111)
        ax1.set_ylim = ([0.9,1])
        ax1.set_xlim = ([0,365])
        kmf_naive = KaplanMeierFitter()
        ax1 = kmf_naive.fit(duration0,C0,label='opioid-').plot()
        ax1.set_title('manipulation revision free survival')
        ax1.set_xlabel('days after arthroplasty')
        ax1.set_ylabel('manipulation free probability')
        kmf_chronic= KaplanMeierFitter()
        ax1 = kmf_chronic.fit(duration1,C1,label='opioid+').plot(ax=ax1,color='r').plot()
        plt.show
        plt.savefig('KM_mnp.pdf')
        summary= logrank_test(duration1,duration0,C1,C0,alpha=0.99)
    #
        

def plot_KM_rev(w5):
    
    """"A method to make KM plots  it returns a plot of the two graphs and the logrank test result"""
    opioid = w5[(w5.chronic==1)]
    non_opioid = w5[~(w5.chronic==1)]
    #
    duration0 = non_opioid.t2rev
    duration0= duration0.ix[duration0>0]

    duration1 = opioid.t2rev
    duration1 = duration1.ix[duration1>0]
	#
    C1 = opioid.RevLogical
    C1=C1.ix[duration1.index]
    C0 = non_opioid.RevLogical
    C0 = C0.ix[duration0.index]
    #
    ax1 = plt.subplot(111)
    ax1.set_ylim = ([0.9,1])
    ax1.set_xlim = ([0,365])
    kmf_naive = KaplanMeierFitter()
    ax1 = kmf_naive.fit(duration0,C0,label='opioid-').plot()
    ax1.set_title('knee revision free survival')
    ax1.set_xlabel('days after arthroplasty')
    ax1.set_ylabel('revision free probability')
    kmf_chronic= KaplanMeierFitter()
    ax1 = kmf_chronic.fit(duration1,C1,label='opioid+').plot(ax=ax1,color='r')
    plt.show
    plt.savefig('KM_rev.pdf')
    summary,p_val, test= logrank_test(duration1,duration0,C1,C0,alpha=0.99)
    return summary
####
####
