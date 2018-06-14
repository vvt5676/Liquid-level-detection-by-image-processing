import cv2
import numpy as np


T_R = -1
T_B = -1
T_G = -1
E_T_R = -1
E_T_B = -1
E_T_G = -1
apr_r = -1
apr_b = -1
apr_g = -1
hsv = -1
lower = []
upper = []
std_height = 1800
def draw_line(canny_edges, img):
    
    lines = cv2.HoughLines(canny_edges,1,np.pi/180,100)
    min_l = np.pi/2 - 0.005
    max_l = np.pi/2 + 0.005
    print(lines.shape)
    for line in lines:
        rho = line[0][0]
        theta =  line[0][1]
        if theta <= max_l and theta >= min_l:
            
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    return img

def sobel_x(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    cv2.imshow("img_x", sobelx)
    return sobelx



def dilate_image(img):
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(img,kernel,iterations = 1)
    return dilate

def crop_bottle(image):
    #sobelx = cv2.Sobel(image,cv2.CV_64F, 1, 0, ksize = 1)
    #sobely = cv2.Sobel(image,cv2.CV_64F, 0, 1, ksize = 1)
    cv2.imshow("laplacian", image)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    canny_edges = cv2.Canny(image, 0, 200)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(laplacian,kernel,iterations = 1)
    dilation = np.uint8(dilation)
    #canny_edges = cv2.Canny(dilation, 10, 100)
    cv2.imshow("laplacian", laplacian)
    #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    new_contours = []
    (h, w) = image.shape
    img = cv2.imread("bottle5.jpg")
    for contour in contours:
        if cv2.contourArea(contour)>h*w/20:
            new_contours.append(contour)
    
    cv2.drawContours(img, new_contours, -1, (0,0,255), 3)
    print(len(new_contours))
    cv2.imshow("contours for bottle ",img)
    cv2.waitKey()
    cv2.imshow("contours for bottle ",image)
    

def find_color_space(img):
    img = cv2.imread(img)
    b, g, r = cv2.split(img)
    b = np.mean(b)
    g = np.mean(g)
    r = np.mean(r)
    print("BGR values: ")
    print(b, g, r)
    li_color = np.uint8([[[b,g,r ]]])
    hsv = cv2.cvtColor(li_color,cv2.COLOR_BGR2HSV)
    print("HSV values")
    print(hsv)
    return hsv


def crop_bottle_vvt(image_name):
    image = cv2.imread(image_name,0)
    
    #laplacian = cv2.Laplacian(image, cv2.CV_64F)
    canny_edges = cv2.Canny(image, 0, 20)
    
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(canny_edges,kernel,iterations = 4)
    dilation = np.uint8(dilation)
    (h, w) = dilation.shape
    print(str(h)+" "+str(w))
    for i in range(w):
        dilation[h-3,i] = 255
        dilation[h-2,i] = 255
        dilation[h-1,i] = 255
        
    cv2.imwrite("dilated.jpg", dilation)
    #canny_edges = cv2.Canny(dilation, 10, 100)
    cv2.imshow("dilated", dilation)
    #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    new_contours = []
    (h, w) = image.shape
    img = cv2.imread(image_name)
    for contour in contours:
        if cv2.contourArea(contour)>h*w/10:
            new_contours.append(contour)
    im2 =  np.zeros((img.shape[0], img.shape[1],3), np.uint8)
    
    cv2.drawContours(img, new_contours, -1, (0,0,255), 3)
    cv2.imwrite("final_image.jpg", img)
    print(len(new_contours))
    return new_contours[0]





def set_intensity_full(img):
    img = cv2.imread(img)
    b, g, r = cv2.split(img)
    T_B = np.mean(b)
    T_G = np.mean(g)
    T_R = np.mean(r)
    return T_B, T_G, T_R


def set_intensity_empty(img):
    img = cv2.imread(img)
    print("inside empty")
    b, g, r = cv2.split(img)
    cv2.imshow("hg", b)
    E_T_B = np.mean(b)
    E_T_G = np.mean(g)
    E_T_R = np.mean(r)
    return  E_T_B, E_T_G, E_T_R

def set_apr_values():
    apr_r = abs(E_T_R - T_R)/2
    apr_b = abs(E_T_B - T_B)/2
    apr_g = abs(E_T_G - T_G)/2
    return apr_b, apr_g, apr_r


    
def level_detection(img, contour):
    
    if(apr_r == -1 or apr_g == -1 or apr_b == -1):
        print("First set the values of the threshold")
        return
    
    img = cv2.imread(img)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.imshow("final contour", cv2.drawContours(img, contour, -1, (0,0,255), 3))
    count_liquid = 0
    count_empty = 0
    total_pixel = 0
    print(x, y, w, h)
    
    for i in range(y, y+h):
        for j in range(x, x+w):
             if cv2.pointPolygonTest(contour, (i, j), False) >0 :
                total_pixel = total_pixel + 1           
                if abs(img[i][j][0]-T_B) <= apr_b and abs(img[i][j][0]-T_B) <= apr_b and abs(img[i][j][0]-T_B) <= apr_b:
                    count_liquid = count_liquid+1
                if abs(img[i][j][0]-E_T_B) <= apr_b and abs(img[i][j][0]-E_T_B) <= apr_b and abs(img[i][j][0]-E_T_B) <= apr_b:
                    count_empty = count_empty+1
    
    res = count_liquid/(count_liquid + count_empty)
    print(count_liquid,count_empty, res)
    return res



#if cv2.pointPolygonTest(contour, (i, j), False) >0 :
                #total_pixel = total_pixel + 1
            
# read the image
def main_function_old():
    global E_T_R, E_T_B, E_T_G
    global T_R, T_B, T_G
    global apr_r, apr_b, apr_g
    
    if E_T_R == -1 or E_T_B == -1 or  E_T_G == -1:
        print("caliberate empty bottle name of the image:")
        image_name = input()
        E_T_R, E_T_B, E_T_G = set_intensity_empty(image_name)
        print("Updated parameters: ", E_T_B, E_T_G, E_T_R)
    if T_R == -1 or T_B == -1 or  T_G == -1:
        
        print("caliberate full bottle name of the image:")
        image_name = input()
        T_B, T_G, T_R = set_intensity_full(image_name)
        print("Updated parameters: ", T_B, T_G, T_R)
    if apr_r == -1:
        apr_r, apr_b, apr_g = set_apr_values()
    print(apr_r, apr_g, apr_b)
    
    while True :
        print("do you have next image:(y/n):")
        string = input()
        if string == "n":
            break
        print("image name to detect the level:")
        image_name = input()
        contour = crop_bottle_vvt(image_name)
        level_detection(image_name, contour)

def set_threshold(sample_image):
    global hsv
    global lower
    global upper
    hsv = find_color_space(sample_image)
    lower = np.array([hsv[0][0][0]-5, hsv[0][0][1]-50, hsv[0][0][2]-50])
    upper = np.array([hsv[0][0][0]+5, hsv[0][0][1]+50, hsv[0][0][2]+50])
    
def check_for_full(image_name):
    global std_height
    
    img = cv2.imread(image_name)
    h, w, c = img.shape
    print(h, w, c)
    std_height = h*80/100
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, lower, upper)
    count = 0
    height = 0;
    for i in range(h):
        for j in  range(w):
            if mask[i][j]>=100:
                count = count+1
            if count>50:
                height = i
                break
        if height != 0 :
            break
    img = cv2.imread(image_name)
    img = cv2.line(img, (0,height), (w,height), (0,0,255),10)
    cv2.imwrite("results.jpg", img)
    if height != 0:
        #print("checking", height)
        height = h-height
    percent = (height/std_height)*100
    print("height filled: ",height)
    print("std_height : ", std_height)
    print("amount of bottle filled: ", percent)
    if percent < 80:
        print("Bottle empty, Skip it ")
    
    
    
    cv2.imwrite("mask.jpg", mask)
    



def main_function():
    global hsv
    if hsv == -1:
        print("Enter the color sample of the liquid\n")
        image_name = input()
        set_threshold(image_name)
    while(True):
        print("Input the name of the image or to exit type 'n'")
        image = input()
        if(image == 'n'):
            break
        check_for_full(image)
main_function()


'''
#convert the image to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# detect the edges with canny edge method

new_image = crop_bottle(img_gray)

#cv2.imshow("dilated image", dilate_image(edges))
#cv2.imshow("before equ",img_gray)
equ = cv2.equalizeHist(img_gray)

#ret, imgf = cv2.threshold(equ, 0, 255,cv2.THRESH_OTSU)
#cv2.imshow("after equilization", equ)  
edges = cv2.Canny(equ,100, 200)
#cv2.imshow("f",edges)
contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
cv2.drawContours(img, contours, -1, (0,255,0), 1)

#cv2.imshow("img with contours", img)

'''
