import os
import sys
import numpy as np
import vtk
from vtk.util import numpy_support as vtk_np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 6})
from matplotlib.colors import LogNorm
from scipy.interpolate import RectBivariateSpline
import math

def loadData(fname,arrName):
	reader = vtk.vtkXMLImageDataReader()  # read vti file
	reader.SetFileName(fname)
	reader.Update()
	data = reader.GetOutput() # load data
	dim = data.GetDimensions() # domain size
	spacing = data.GetSpacing() # resolution 
	n = data.GetCellData().GetArray(arrName).GetNumberOfComponents() # number of compnents
	if (n == 3):
		isVector = 1
	else:
		isVector = 0

	return data, dim, spacing, n, isVector 
	
def getArrayAsImg(data,arrName,v_comp):
	out = data
	dim = out.GetDimensions()
	n = out.GetCellData().GetArray(arrName).GetNumberOfComponents()
	if (n == 3):
		arr = np.array(out.GetCellData().GetArray(arrName))
		arr_x = np.reshape(arr[:,0],(dim[0]-1,dim[1]-1,dim[2]-1),order='F')
		arr_y = np.reshape(arr[:,1],(dim[0]-1,dim[1]-1,dim[2]-1),order='F')
		arr_z = np.reshape(arr[:,2],(dim[0]-1,dim[1]-1,dim[2]-1),order='F')
		if (v_comp == 1):
			arr = arr_x
		elif (v_comp == 2):
			arr = arr_y
		elif (v_comp == 3):
			arr = arr_z
		else:
			arr = np.sqrt(np.power(arr_x,2) + np.power(arr_y,2) + np.power(arr_z,2))
# 		print('vector')
	else:
		arr = np.array(out.GetCellData().GetArray(arrName))
		arr = np.reshape(arr,(dim[0]-1,dim[1]-1,dim[2]-1),order='F')
	
	return arr	


def showMap(ax,data,xlms,ylms,xstp_mm,ystp_mm,spacing,caxlim,clrmap):
    out = data
    #ax = plt.gca()
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    im = ax.imshow(out,norm=LogNorm(vmin=caxlim[0], vmax=caxlim[1]),cmap=clrmap) # Plot image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)

    cbar = plt.colorbar(im,cax=cax)
    #im.set_clim(caxlim[0],caxlim[1]) 
    ax.invert_yaxis()
    #cbar.set_label("$n_e$ [x $10^{18}$ cm$^{-3}$]")
    
    # set limits
    xstp = round(xstp_mm / (spacing[0]*1e3))
    ystp = round(ystp_mm / (spacing[0]*1e3))
    ax.set_xlim([xlms[0] * 1/(spacing[0]*1e3) + out.shape[1]/2, xlms[1] * 1/(spacing[0]*1e3) + out.shape[1]/2] )
    ax.set_ylim([ylms[0]  * 1/(spacing[0]*1e3) + out.shape[0]/2, ylms[1] * 1/(spacing[0]*1e3) + out.shape[0]/2] )

    # Set axis labels

    ax.set_xticks(np.round(np.arange(xlms[0]*1/(spacing[0]*1e3),xlms[1]*1/(spacing[0]*1e3),step=xstp) + out.shape[1]/2))
    ax.set_yticks(np.round(np.arange(ylms[0]*1/(spacing[0]*1e3),ylms[1]*1/(spacing[0]*1e3),step=ystp) + out.shape[0]/2))

    labels_xmm = np.round((ax.get_xticks()-out.shape[1]/2) * spacing[0] *1e3)
    labels_ymm = np.round((ax.get_yticks()-out.shape[0]/2) * spacing[0] *1e3)
    ax.set_xticklabels(labels_xmm)
    ax.set_yticklabels(labels_ymm)
    return cbar, im

def showMapNormal(ax,data,xlms,ylms,xstp_mm,ystp_mm,spacing,caxlim,clrmap):
    out = data
    #ax = plt.gca()
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    im = ax.imshow(out,vmin=caxlim[0], vmax=caxlim[1],cmap=clrmap) # Plot image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)

    cbar = plt.colorbar(im,cax=cax)
    #im.set_clim(caxlim[0],caxlim[1]) 
    ax.invert_yaxis()
    #cbar.set_label("$n_e$ [x $10^{18}$ cm$^{-3}$]")
    
    # set limits
    xstp = round(xstp_mm / (spacing[0]*1e3))
    ystp = round(ystp_mm / (spacing[0]*1e3))
    ax.set_xlim([xlms[0] * 1/(spacing[0]*1e3) + out.shape[1]/2, xlms[1] * 1/(spacing[0]*1e3) + out.shape[1]/2] )
    ax.set_ylim([ylms[0]  * 1/(spacing[0]*1e3) + out.shape[0]/2, ylms[1] * 1/(spacing[0]*1e3) + out.shape[0]/2] )

    # Set axis labels

    ax.set_xticks(np.round(np.arange(xlms[0]*1/(spacing[0]*1e3),xlms[1]*1/(spacing[0]*1e3),step=xstp) + out.shape[1]/2))
    ax.set_yticks(np.round(np.arange(ylms[0]*1/(spacing[0]*1e3),ylms[1]*1/(spacing[0]*1e3),step=ystp) + out.shape[0]/2))

    labels_xmm = np.round((ax.get_xticks()-out.shape[1]/2) * spacing[0] *1e3)
    labels_ymm = np.round((ax.get_yticks()-out.shape[0]/2) * spacing[0] *1e3)
    ax.set_xticklabels(labels_xmm)
    ax.set_yticklabels(labels_ymm)
    return cbar,im

def showMapNormalV2(ax,data,ogn,xlms,ylms,xstp_mm,ystp_mm,spacing,caxlim,clrmap):
    # spacing - m/px
    # Adding ogn 
    out = data
    #ax = plt.gca()
    im = ax.imshow(out,vmin=caxlim[0], vmax=caxlim[1],cmap=clrmap) # Plot image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im,cax=cax)
    ax.invert_yaxis()
    
    # set limits
    xstp = round(xstp_mm / (spacing[0]*1e3))
    ystp = round(ystp_mm / (spacing[0]*1e3))
    ax.set_xlim([xlms[0] * 1/(spacing[0]*1e3) + ogn[0], xlms[1] * 1/(spacing[0]*1e3) + ogn[0]] )
    ax.set_ylim([ylms[0]  * 1/(spacing[0]*1e3) + ogn[1], ylms[1] * 1/(spacing[0]*1e3) + ogn[1]] )

    # Set axis labels

    ax.set_xticks(np.round(np.arange(xlms[0]*1/(spacing[0]*1e3),xlms[1]*1/(spacing[0]*1e3)+xstp,step=xstp) + ogn[0]))
    ax.set_yticks(np.round(np.arange(ylms[0]*1/(spacing[0]*1e3),ylms[1]*1/(spacing[0]*1e3)+ystp,step=ystp) + ogn[1]))

    labels_xmm = np.round((ax.get_xticks()-ogn[0]) * spacing[0] *1e3)
    labels_ymm = np.round((ax.get_yticks()-ogn[1]) * spacing[0] *1e3)
    ax.set_xticklabels(labels_xmm)
    ax.set_yticklabels(labels_ymm)
    
    # Grid
    x = np.linspace(0,data.shape[1],num=data.shape[1]) # [px]
    y = np.linspace(0,data.shape[0],num=data.shape[0]) # [px]
    xx, yy = np.meshgrid(x,y)
    xx = (xx - ogn[0]) * spacing[0] * 1e3 # [mm]
    yy = (yy - ogn[1]) * spacing[0] * 1e3 # [mm] 
    
    return xx,yy,cbar,im

def showMapV2(ax,data,ogn,xlms,ylms,xstp_mm,ystp_mm,spacing,caxlim,clrmap):
    # Adding ogn 
    out = data
    #ax = plt.gca()
    im = ax.imshow(out,norm=LogNorm(vmin=caxlim[0], vmax=caxlim[1]),cmap=clrmap) # Plot image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im,cax=cax)
    ax.invert_yaxis()
    
    # set limits
    xstp = round(xstp_mm / (spacing[0]*1e3))
    ystp = round(ystp_mm / (spacing[0]*1e3))
    ax.set_xlim([xlms[0] * 1/(spacing[0]*1e3) + ogn[0], xlms[1] * 1/(spacing[0]*1e3) + ogn[0]] )
    ax.set_ylim([ylms[0]  * 1/(spacing[0]*1e3) + ogn[1], ylms[1] * 1/(spacing[0]*1e3) + ogn[1]] )

    # Set axis labels

    ax.set_xticks(np.round(np.arange(xlms[0]*1/(spacing[0]*1e3),xlms[1]*1/(spacing[0]*1e3)+xstp,step=xstp) + ogn[0]))
    ax.set_yticks(np.round(np.arange(ylms[0]*1/(spacing[0]*1e3),ylms[1]*1/(spacing[0]*1e3)+ystp,step=ystp) + ogn[1]))

    labels_xmm = np.round((ax.get_xticks()-ogn[0]) * spacing[0] *1e3)
    labels_ymm = np.round((ax.get_yticks()-ogn[1]) * spacing[0] *1e3)
    ax.set_xticklabels(labels_xmm)
    ax.set_yticklabels(labels_ymm)
    
    # Grid
    x = np.linspace(0,data.shape[1],num=data.shape[1]) # [px]
    y = np.linspace(0,data.shape[0],num=data.shape[0]) # [px]
    xx, yy = np.meshgrid(x,y)
    xx = (xx - ogn[0]) * spacing[0] * 1e3 # [mm]
    yy = (yy - ogn[1]) * spacing[0] * 1e3 # [mm] 
    
    return xx,yy,cbar,im

def getAvgSlice(data,zid):
    return (data[:,:,int(zid)] + data[:,:,int(zid+1)] + data[:,:,int(zid-1)])/3

def getAvgLineoutY(spl,Xq,Yq,stp,N):
    Y = np.linspace(-stp*N,stp*N,num=2*N+1) # we wantto average over Yq +/- st*N w N points
    A = np.zeros(Xq.shape)
    for ii in range(Y.size):
#         print(Y[ii])
        Ynow = Yq + Y[ii]
        A = np.add(A,spl.ev(Xq,Ynow))
    return A / (2*N+1)


def getAzmuthalAvg(spl,xx,yy,Rq,res):
    th = np.linspace(0,1*math.pi,num=res)
    out = []; sd = []
    for ii in range(Rq.size):
        Xq = Rq[ii] * np.cos(th)
        Yq = Rq[ii] * np.sin(th)
        out.append(np.mean(spl.ev(Xq,Yq)))
        sd.append(np.std(spl.ev(Xq,Yq)))
        
    return np.asarray(out), np.asarray(sd)

def getAzmuthalAvg2(spl,xx,yy,Rq,x0,y0,a1,a2,res):
    th = np.linspace(a1,a2,num=res)
    out = []; sd = []
    for ii in range(Rq.size):
        Xq = Rq[ii] * np.cos(th) + x0
        Yq = Rq[ii] * np.sin(th) + y0
        out.append(np.mean(spl.ev(Xq,Yq)))
        sd.append(np.std(spl.ev(Xq,Yq)))
        
    return np.asarray(out), np.asarray(sd)
        
        
    