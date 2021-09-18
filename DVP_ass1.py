import pandas as pd
import cv2
import glob
from scipy import signal
from matplotlib import pyplot as plt
import regex as re
import numpy as np
import scipy.ndimage
from skimage.metrics import structural_similarity as ssim
plt.rcParams['figure.dpi'] = 1000



#####################################################################################################
# Before running the code put all the images in one folder (i.e all corridor and sphere images in one folder and specify it in "path" variable in main() function
#####################################################################################################
#%%
def multi_scale_horn_schunk(f1,f2,LAMBDA,tol,scale):
    
    h,w         =   f1.shape
    SCALES      =   [(int(h/2**i),int(w/2**i)) for i in range(0,scale)][::-1]
    # print(SCALES[0])
    
    f1_resized  =   [cv2.resize(f1,SCALES[i],interpolation=cv2.INTER_CUBIC) for i in range(len(SCALES))]
    f2_resized  =   [cv2.resize(f2,SCALES[i],interpolation=cv2.INTER_CUBIC) for i in range(len(SCALES))]
    
    
    a_final , b_final     = horn_schunk(f1_resized[0] ,f2_resized[0], LAMBDA , tol)
    for i in range(1,len(SCALES)):
        a_resized , b_resized = cv2.resize(a_final,SCALES[i]) , cv2.resize(b_final,SCALES[i])
        img_wrapp = image_warping(f1_resized[i],a_resized,b_resized)
        a_curr , b_curr = horn_schunk(img_wrapp  ,f2_resized[i], LAMBDA , tol)
        a_final , b_final = a_resized+a_curr , b_resized +b_curr 
    
    return a_final , b_final

#%%
def multi_scale_lucas_canade(f1,f2,patch_size,scale):
    
    h,w         =   f1.shape
    SCALES      =   [(int(h/2**i),int(w/2**i)) for i in range(0,scale)][::-1]
    
    patches     =   [int(patch_size/2**i) for i in range(0,scale)][::-1]
    
    f1_resized  =   [cv2.resize(f1,SCALES[i],interpolation=cv2.INTER_CUBIC) for i in range(len(SCALES))]
    f2_resized  =   [cv2.resize(f2,SCALES[i],interpolation=cv2.INTER_CUBIC) for i in range(len(SCALES))]
    
    # print(type(patches[0])) 
    a_final , b_final     = lucas_canade(f1_resized[0] ,f2_resized[0],patches[0])
    
    for i in range(1,len(SCALES)):
        a_resized , b_resized = cv2.resize(a_final,SCALES[i]) , cv2.resize(b_final,SCALES[i])
        img_wrapp = image_warping(f1_resized[i],a_resized,b_resized)
        a_curr , b_curr = lucas_canade(img_wrapp,f2_resized[i],patches[i])
        a_final , b_final = a_resized+a_curr , b_resized +b_curr 
    
    return a_final , b_final



#%%
def horn_schunk(f1,f2,lamb,tol):
    
    h,w=f1.shape
    
    # Intializing optical flow. 
    a=np.zeros((h,w))
    b=np.zeros((h,w))
    
    # print(lamb)
    
    # average optical flow to compute laplacian = a_avg-a
    a_avg=np.zeros((h,w))
    b_avg=np.zeros((h,w))
    
    
    # filter_x    = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # filter_y    = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    # filter_t    = np.array([[-1,-1],[-1,-1]])
    

    
    filter_x    = np.array([[-1,1],[-1,1]])
    filter_y    = np.array([[-1,-1],[1,1]])
    filter_t    = np.array([[-1,-1],[-1,-1]])
    
    filter_avg  = np.array([[0, 1 / 4, 0], [1 / 4, 0, 1 / 4], [0, 1 / 4, 0]], dtype=np.float32)
    # filter_avg    = np.array([[1/12,1/6,1/12],
    #                   [1/6,-1,1/6],
    #                 [1/12,1/6,1/12]])

    
    f_x=(cv2.filter2D(f1,-1,filter_x)+cv2.filter2D(f2,-1,filter_x))*0.5
    f_y=(cv2.filter2D(f1,-1,filter_y)+cv2.filter2D(f2,-1,filter_y))*0.5
    f_t=(cv2.filter2D(f1,-1,filter_t)+cv2.filter2D(f2,-1,-1*filter_t))
    
    
    grad_norm=f_x**2+f_y**2
    
    a_curr=a
    b_curr=b
    

    
    while(True):
        a_prev,b_prev = a_curr,b_curr   
        a_avg , b_avg = cv2.filter2D(a_prev,-1,filter_avg) , cv2.filter2D(b_prev,-1,filter_avg)
        a_curr        = a_avg-lamb*f_x*((f_x*a_avg + f_y*b_avg + f_t)/(1+lamb*grad_norm))
        b_curr        = b_avg-lamb*f_y*((f_x*a_avg + f_y*b_avg + f_t)/(1+lamb+grad_norm))
        if(abs((a_curr-a_prev)).max() and abs((b_curr-b_prev)).max()):
            break

    return  a_curr ,b_curr    

#%%

def show_op_flow(frame,a,b):
    fig=plt.figure(dpi=500)
    for i in range(0,frame.shape[0],6):
        for j in range(0,frame.shape[1],6):
            plt.arrow(j,i,a[i,j],b[i,j],color='w')#head_width=2)
           
    plt.imshow(255*np.zeros(frame.shape),cmap='gray')
    plt.show()
    return

    
#%%

def image_warping(frame,a,b):
    x,y=np.float32(np.meshgrid(np.arange(frame.shape[0]),np.arange(frame.shape[1])))
    x1,y1=np.float32(x+a),np.float32(y+b)
    warped_image=cv2.remap(frame,x1,y1,interpolation=cv2.INTER_CUBIC)    
    return warped_image


#%%
def compute_ssim(interpolated,ground_truth):
    ground_truth   =   ground_truth/ground_truth.max()
    interpolated  =   interpolated/interpolated.max()
    ground_truth   =   ground_truth .astype('float32')
    interpolated  =   interpolated.astype('float32')        
    
    
    return ssim(ground_truth,interpolated)




#%%
def sphere_interpol(path,method):
    path=path+"\*.ppm"
    images = glob.glob(path)
    images.sort(key=lambda f: int(re.sub("\D", "", f)))
   
    
    
    sphere_images=[cv2.imread(images[i], flags=cv2.IMREAD_GRAYSCALE) for i in range(len(images))]
    
    # Parameters for horn_schunk.
    LAMBDA=0.2
    tol=0.001
    scale=1
    
    # Paramters for lucas_canade
    
    patch_size = 7
    
    for i in range(len(images)-2):
        f1            =  np.array(sphere_images[i],dtype=np.float32)        # First Frame
        
        f2            =  np.array(sphere_images[i+2],dtype=np.float32)      # Second Frame
       
        f_ground      =  np.array(sphere_images[i+1],dtype=np.float32)      # Ground Frame

        
        if(method=="horn_schunk"):
            a_forw ,b_forw  = multi_scale_horn_schunk (f1,f2,LAMBDA,tol,scale)
            
            interpolated_forw = image_warping(f1,a_forw,b_forw)
            show_op_flow(f1,a_forw ,b_forw)
            
            a_back ,b_back  = multi_scale_horn_schunk (f2,f1,LAMBDA,tol,scale)
            
            show_op_flow(f2,a_back ,b_back)
            
            interpolated_back = image_warping(f2,a_back,b_back)
            
            interpolated_frame = (interpolated_forw+interpolated_back)/2
            # print(interpolated_frame)
            SSIM    =    compute_ssim(interpolated_frame , f_ground)
            plt.imshow(interpolated_frame,cmap='gray')
            plt.show()
            
            print(f"SSIM using horn schunk for sphere."+str(i+1)+" is",SSIM)
        else:
            a_forw ,b_forw  = multi_scale_lucas_canade(f1,f2,patch_size,scale)
            
            show_op_flow(f1,a_forw ,b_forw)
            interpolated_forw = image_warping(f1,a_forw,b_forw)
        
            a_back ,b_back  = multi_scale_lucas_canade(f2,f1,patch_size,scale)
            
            show_op_flow(f2,a_back ,b_back)
            interpolated_back = image_warping(f2,a_back,b_back)
            
            interpolated_frame = (interpolated_forw+interpolated_back)/2
            
            SSIM    =    compute_ssim(interpolated_frame , f_ground)
            
            plt.imshow(interpolated_frame,cmap='gray')
            plt.show()
            
            print(f"SSIM using lucas canade for sphere."+str(i+1)+" is",SSIM)

        # show_op_flow(f2,a_back ,b_back)
#%%
def corridor_interpol(path,method):
    path=path+"\*.pgm"
    images = glob.glob(path)
   
    images.sort(key=lambda f: int(re.sub("\D", "", f)))
    corridor_images=[cv2.imread(images[i], flags=cv2.IMREAD_GRAYSCALE) for i in range(len(images))]
    
    
    # Parameters for horn_schunk.
    LAMBDA=0.002
    tol=0.001
    scale=3
    
    # Paramters for lucas_canade
    
    patch_size = 5
    
    
    
    
    for i in range(len(images)-2):
        f1            =  np.array(corridor_images[i],dtype=np.float32)        # First Frame
        
        print(f1.shape)
        
        f2            =  np.array(corridor_images[i+2],dtype=np.float32)      # Second Frame
       
        f_ground      =  np.array(corridor_images[i+1],dtype=np.float32)      # Ground Frame

        
        if(method=="horn_schunk"):
            a_forw ,b_forw  = multi_scale_horn_schunk (f1,f2,LAMBDA,tol,scale)
            
            interpolated_forw = image_warping(f1,a_forw,b_forw)
            show_op_flow(f1,a_forw ,b_forw)
            
            a_back ,b_back  = multi_scale_horn_schunk (f2,f1,LAMBDA,tol,scale)
            
            show_op_flow(f1,a_forw ,b_forw)
            interpolated_back = image_warping(f2,a_back,b_back)
            
            interpolated_frame = (interpolated_forw+interpolated_back)/2
            
            SSIM    =    compute_ssim(interpolated_frame , f_ground)
            
            plt.imshow(interpolated_frame,cmap='gray')
            plt.show()
            
            print(f"SSIM using horn schunk for sphere."+str(i+1)+" is",SSIM)
        else:
            a_forw ,b_forw  = multi_scale_lucas_canade(f1,f2,patch_size,scale)
            
            show_op_flow(f1,a_forw ,b_forw)
            
            interpolated_forw = image_warping(f1,a_forw,b_forw)
        
            a_back ,b_back  = multi_scale_lucas_canade(f2,f1,patch_size,scale)
            
            
            show_op_flow(f1,a_forw ,b_forw)
            interpolated_back = image_warping(f2,a_back,b_back)
            
            interpolated_frame = (interpolated_forw+interpolated_back)/2
            
            plt.imshow(interpolated_frame,cmap='gray')
            plt.show()
            
            SSIM    =    compute_ssim(interpolated_frame , f_ground)
            
            print(f"SSIM using lucas canade for sphere."+str(i+1)+" is",SSIM)
        
#%%
def lucas_canade(f1,f2,patch_size):
    h,w=f1.shape
    
    # Intializing optical flow. 
    a=np.zeros((h,w))
    b=np.zeros((h,w))
    
    
    filter_x    = np.array([[-1,1],[-1,1]])
    filter_y    = np.array([[-1,-1],[1,1]])
    filter_t    = np.array([[-1,-1],[-1,-1]])
    
    f_x=(cv2.filter2D(f1,-1,filter_x)+cv2.filter2D(f2,-1,filter_x))*0.5
    f_y=(cv2.filter2D(f1,-1,filter_y)+cv2.filter2D(f2,-1,filter_y))*0.5
    f_t=(cv2.filter2D(f1,-1,filter_t)+cv2.filter2D(f2,-1,-1*filter_t))
    
    s  = int(patch_size/2) 
    
    for i in range(s,h-s):
        for j in range(s,w-s):
            Ix = f_x[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = f_y[i-w:i+w+1, j-w:j+w+1].flatten()
            It = f_t[i-w:i+w+1, j-w:j+w+1].flatten()
            
            B= np.reshape(It, (It.shape[0], 1)) 
            A=np.vstack((Ix, Iy)).T  
            op=np.matmul(np.linalg.pinv(A+0.001), B) 

            a[i, j]=op[0]
            b[i, j]=op[1]
            
            print
    
            
    
        
    return a ,b      

            
    
#%%
def main():
    
    # Define path here
    
    
    path=r"C:/Users/Smart/Desktop/Assignment/sphere"
    
    # Parameters for sphere interpolation


    # sphere_interpol(path,"horn_schunk")
    # sphere_interpol(path,"lucas_canade")
    # corridor_interpol(path,"horn_schunk")
    corridor_interpol(path,"lucas_canade")
  # 
#%%
if __name__ == "__main__":
    main()