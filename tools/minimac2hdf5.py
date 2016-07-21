
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PYTHON_PATH
if PYTHON_PATH is not None:
	for i in PYTHON_PATH: sys.path.insert(0,i)
import argparse
import h5py
import pandas as pd
import numpy as np
from hdgwas.tools import Timer
import tables



_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
		  'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
	'''Private.
    '''
	global _proc_status, _scale
	# get pseudo file  /proc/<pid>/status
	try:
		t = open(_proc_status)
		v = t.read()
		t.close()
	except:
		return 0.0  # non-Linux?
		# get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
	i = v.index(VmKey)
	v = v[i:].split(None, 3)  # whitespace
	if len(v) < 3:
		return 0.0  # invalid format?
		# convert Vm value to bytes
	return float(v[1]) * _scale[v[2]]


def memory(since=0.0):
	'''Return memory usage in bytes.
    '''
	return _VmB('VmSize:') - since


def resident(since=0.0):
	'''Return resident memory usage in bytes.
    '''
	return _VmB('VmRSS:') - since


def probes_minimac2hdf5(data_path, save_path,study_name):
	#raise ValueError('test')
	print 'memory before read probes', memory()
	n=[]


	df=pd.read_csv(data_path,sep=' ',chunksize=500000, header=None,index_col=None)

	for i,chunk in enumerate(df):
		print 'add chunk {}'.format(i)
		chunk.columns=["ID",'allele1','allele2','MAF','Rsq']
		chunk.to_hdf(os.path.join(save_path,'probes',study_name+'.h5'), key='probes',format='table',append=True,
				 min_itemsize = 25, complib='zlib',complevel=9 )

	# f=open(data_path,'r')
	# for i,j in enumerate(f):
	# 	n.append((j[:-1]).split(' '))
	# 	if i>=500000 and i%500000==0:
	# 		print 'add chunk {}'.format(str(i/500000) )
	# 		n=np.array(n)
	# 		print n.shape
	# 		chunk=pd.DataFrame.from_dict({"ID":n[:,0],'allele1':n[:,1],'allele2':n[:,2],'MAF':n[:,3],'Rsq':n[:,4]})
	# 		n=[]
	# 		chunk.to_hdf(os.path.join(save_path,'probes',study_name+'.h5'), key='probes',format='table',append=True,
	# 			 min_itemsize = 25, complib='zlib',complevel=9 )
    #
	# f.close()
	# print 'memory after read probes', memory()
	# n=np.array(n)
	# print 'memory after probes2npy', memory()
	# chunk=pd.DataFrame.from_dict({"ID":n[:,0],'allele1':n[:,1],'allele2':n[:,2],'MAF':n[:,3],'Rsq':n[:,4]})
	# print 'memory after dic', memory()
	# chunk.to_hdf(os.path.join(save_path,'probes',study_name+'.h5'), key='probes',format='table',append=True,
	# 			 min_itemsize = 25, complib='zlib',complevel=9 )

def ind_minimac2hdf5(data_path, save_path,study_name):
	n=[]
	f=open(data_path,'r')
	for i,j in enumerate(f):
		n.append((j[:-1]))
	f.close()
	n=np.array(n)
	chunk=pd.DataFrame.from_dict({"individual":n})
	chunk.to_hdf(os.path.join(save_path,'individuals',study_name+'.h5'), key='individuals',format='table',
				 min_itemsize = 25, complib='zlib',complevel=9 )

def id_minimac2hdf5(data_path,id, save_path):

	n=[]
	f=open(data_path,'r')
	for i,j in enumerate(f):
		try:
			n.append(np.float(j))
		except:
			n.append(np.float(-1))
	n=np.array(n)
	f.close()
	store=h5py.File(os.path.join(save_path,'genotype',id+'.h5'), 'w')
	with Timer() as t:
		store.create_dataset(id,data=n,compression='gzip',compression_opts=9 )
	print 'standard save gzip 9...', t.secs
	store.close()



def id_minimac2hdf5_pandas(data_path,id, save_path):

	df=pd.read_csv(data_path, header=None, index_col=None)
	df.columns=["genotype"]
	n=df["genotype"].as_matrix()
	store=h5py.File(os.path.join(save_path,'genotype',id+'.h5'), 'w')
	with Timer() as t:
		store.create_dataset(id,data=n,compression='gzip',compression_opts=9 )
	print 'pandas save gzip 9...', t.secs
	store.close()
	df=None

def genotype_minimac2hdf5(data_path,id, save_path, study_name):

	df=pd.read_csv(data_path, header=None, index_col=None,sep='\t', dtype=np.float16)
	data=df.as_matrix()
	data=data.T
	print data.shape
	print 'Saving chunk...{}'.format(os.path.join(save_path,'genotype',str(id)+'_'+study_name+'.h5'))
	h5_gen_file = tables.openFile(
		os.path.join(save_path,'genotype',str(id)+'_'+study_name+'.h5'), 'w', title=study_name)

	atom = tables.Float16Atom()
	genotype = h5_gen_file.createCArray(h5_gen_file.root, 'genotype', atom,
										(data.shape),
										title='Genotype',
										filters=tables.Filters(complevel=9, complib='zlib'))
	genotype[:] = data
	h5_gen_file.close()
	os.remove(data_path)


if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Script to convert Minimac data')
	parser.add_argument("-study_name", required=True, type=str, help="Study specific name")
	parser.add_argument("-id", type=str, help="subject id")
	parser.add_argument("-data",required=True, type=str, help="path to file")
	parser.add_argument("-out",required=True, type=str, help="path to results save folder")
	parser.add_argument("-flag",required=True,type=str,choices=['genotype','individuals','probes','chunk'], help="path to file with SNPs info")


	args = parser.parse_args()

	print args
	try:
		print ('Creating directories...')
		os.mkdir(os.path.join(args.out,'genotype') )
		os.mkdir(os.path.join(args.out,'individuals') )
		os.mkdir(os.path.join(args.out,'probes') )
		os.mkdir(os.path.join(args.out,'tmp_files'))
	except:
		print('Directories "genotype","probes","individuals" are already exist in {}...'.format(args.out))

	if args.id is not None and args.flag=='genotype':
		with Timer() as t:
			id_minimac2hdf5_pandas(args.data, args.id, args.out)
		print 'time pandas...',t.secs
	elif args.flag=='probes':
		probes_minimac2hdf5(args.data, args.out, args.study_name)
	elif args.flag=='individuals':
		ind_minimac2hdf5(args.data, args.out,args.study_name)
	elif args.flag=='chunk':
		genotype_minimac2hdf5(args.data,args.id, args.out,args.study_name)



