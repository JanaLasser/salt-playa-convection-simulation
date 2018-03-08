import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import cPickle as pickle

def PrintParams():
	pass

def PrintColorMatrix(Matrix, length, par, vmin = 0., vmax = 0., savepath = '' , savename = 'test', time = 0.):
#	with PdfPages(SAVEPATH+savename +'.pdf') as pdf:
	fig, ax = plt.subplots()
	if vmin != vmax:
		im = ax.imshow(np.transpose(Matrix), cmap=plt.get_cmap('hot'), interpolation='nearest',
   	           vmin=vmin, vmax=vmax, extent=[0.,length[0],length[1],0], aspect = 'equal')
	else:
		im = ax.imshow(np.transpose(Matrix), cmap=plt.get_cmap('hot'), interpolation='nearest', extent=[0.,length[0],length[1],0], aspect = 'auto')
	fig.colorbar(im)
	plt.title('Rayleigh = %.1f, time = %.2f' % (par['Ra'], time))
	plt.xlabel('x-value')
	plt.ylabel('y-value')
#	pdf.savefig()
	plt.savefig(savepath+savename + str(round(time, 2)) + '.png')
	plt.close()

def PrintCrossSection(Row, vmin = 0., vmax = 0., \
	savepath = '',  savename = 'test', time = 0., xlabel = 'x', ylabel = 'y'):
	if vmin != vmax:
		plt.axis([0, np.shape(Row)[0]-1, vmin, vmax])
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(Row)
	plt.savefig(savepath+savename + str(round(time, 2)) + '.png')
	plt.close()

def PlotField(field, time, field_name, savepath):
	savename = '{}_{:1.3f}'.format(field_name, time)
	fig, ax = plt.subplots()
	im = ax.imshow(np.transpose(field), cmap=plt.get_cmap('hot'), \
		interpolation='nearest', aspect = 'auto')
	fig.colorbar(im)
	plt.title('field: {} time: {:1.3f}'.format(field_name, time))
	plt.xlabel('depth [natural units]')
	plt.ylabel('width [natural units]')
	plt.savefig(join(savepath,savename) + '.png')
	plt.close()


def PrintField(field, time, field_name, savepath):
	savename = '{}_{:1.3f}'.format(field_name, time)
	np.save(join(savepath,savename), field)

def CountNumberOfMaxima(Matrix):
	size = np.shape(Matrix)
	number_of_maxima = np.zeros(size[1], dtype=np.int)
	for y in range(size[1]):
		for x in range(size[0]):
			if Matrix[x,y] - 1e-6 > np.max((Matrix[(x-1)%size[0],y],Matrix[(x+1)%size[0],y])):
				number_of_maxima[y] += 1
	return number_of_maxima

# parameters dict example: {'Ra': RA, 'Ra2' : 0., 'A': A, \
#	'amplitude': amplitude, 'waves': waves, 'phi': 0.0, \
#	'max_T':MAXTIME, 'clf':adaptive_dt_constant, 'res':res,\
#	'HEIGHT':HEIGHT, 'LENGHT':LENGHT}
def PrintParams(params, savepath, run_name):
	# use `pickle.loads` to do the reverse
	param_file = open(join(savepath,run_name + '_params.txt'),'a')
	param_file.write('Ra\t{}\n'.format(params['Ra']))
	param_file.write('Ra2\t{}\n'.format(params['Ra2']))
	param_file.write('amplitude\t{}\n'.format(params['amplitude']))
	param_file.write('waves\t{}\n'.format(params['waves']))
	param_file.write('phi\t{}\n'.format(params['phi']))
	param_file.write('max_T\t{}\n'.format(params['max_T']))
	param_file.write('clf\t{}\n'.format(params['clf']))
	param_file.write('res\t{}\n'.format(params['res']))
	param_file.write('HEIGHT\t{}\n'.format(params['HEIGHT']))
	param_file.write('LENGTH\t{}\n'.format(params['LENGTH']))
	param_file.write('seed\t{}\n'.format(params['seed']))

	param_file.close()

	
