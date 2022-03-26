import cv2
import numpy as np
from matplotlib import pyplot as plt




img = cv2.imread('tom and jerry.jpg',)
a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.show()




#Prewitt
def Prewitt(img):
    kernelx = np.array([[1,1,1],[0,0,0,],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    PrewittX = cv2.filter2D(img, -1, kernelx)
    PrewittY = cv2.filter2D(img, -1, kernely)

    plt.imshow(PrewittX,cmap = 'gray')
    plt.title('PrewittX')
    plt.show()
    plt.imshow(PrewittY,cmap = 'gray')
    plt.title('PrewittY')
    plt.show()
    plt.imshow(PrewittX + PrewittY,cmap = 'gray')
    plt.title('Prewitt')
    plt.show()
    cv2.imwrite("New after Prewitt.jpg", PrewittX + PrewittY)





#Sobel
def Sobel(img):
    SobelX = cv2.Sobel(img,cv2.CV_64F,1,0)  
    SobelY = cv2.Sobel(img,cv2.CV_64F,0,1)  

    plt.imshow(SobelX,cmap = 'gray')
    plt.title('SobelX')
    plt.show()
    plt.imshow(SobelY,cmap = 'gray')
    plt.title('SobelY')
    plt.show()
    plt.imshow(SobelX + SobelY,cmap = 'gray')
    plt.title('Sobel')
    plt.show()
    cv2.imwrite("New after Sobel.jpg", SobelX + SobelY)




#Laplacian
def Laplacian(img):
    Laplacian = cv2.Laplacian(img,cv2.CV_64F)
    
    plt.imshow(Laplacian,cmap = 'gray')
    plt.title('Laplacian')
    plt.show()
    cv2.imwrite("New after Laplacian.jpg", Laplacian)




#Canny
def Canny(img, weak_th = None, strong_th = None):      
    
    # Original
    plt.imshow(img,cmap = 'gray')
    plt.title('Original')
    plt.show()        
      
    # Noise reduction
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
    plt.imshow(img,cmap = 'gray')
    plt.title('After Noise reduction')
    plt.show()
    cv2.imwrite("New after Noise reduction.jpg", img)
      
    # Compute gradient
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)  
             
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
    mag_max = np.max(mag)
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
       
    height, width = img.shape
      
    plt.imshow(mag,cmap = 'gray')
    plt.title('After Compute gradient')
    plt.show()
    cv2.imwrite("New after Compute gradient.jpg", mag)
    
    # selecting the neighbours
    for i_x in range(width):
        for i_y in range(height):
               
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)

            # In the x axis direction
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
              
            # top right direction
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
              
            # In y-axis direction
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
              
            # top left direction
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
              
            # restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
               
            # Non-maximum suppression step
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x]= 0
                    continue
   
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x]= 0
                
    
    plt.imshow(mag,cmap = 'gray')
    plt.title('After non-maxima suppression')
    plt.show()
    cv2.imwrite("New after non-maxima suppression.jpg", mag)
    
    # double thresholding step
    ids = np.zeros_like(img)
    for i_x in range(width):
        for i_y in range(height):
              
            grad_mag = mag[i_y, i_x]
              
            if grad_mag<weak_th:
                mag[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th:
                ids[i_y, i_x]= 1
            else:
                ids[i_y, i_x]= 2
           
    plt.imshow(mag,cmap = 'gray')
    plt.title('After double thresholding')
    plt.show()
    cv2.imwrite("New after double thresholding.jpg", mag)
     
     


f1 = Prewitt(a)
f2 = Sobel(a)
f3 = Laplacian(a)
f4 = Canny(a)

