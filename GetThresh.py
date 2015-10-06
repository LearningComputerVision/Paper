'''
Created on Sep 5, 2015

@author: Anh
'''
import cv2
import cv2.cv as cv
import numpy as np
import scipy.io as sio
import scipy
import glob,os
import sys
import time
from matplotlib import pyplot as plt
from theano.scalar.basic import switch
'''
###########################################
##Foreach the image folder of hand
###########################################
'''
def detectHandFolder(in_folder):
    folder_image_test=in_folder+"\\images";
    folder_image_annotation=in_folder+"\\annotations";
    list_image_file=[]
    list_annotation_file=[]
    list_name_image_file=[]
    for root, dirname, file in os.walk(folder_image_test): 
        for filename in file:
            filepath=os.path.join(root,filename)
            list_image_file.append(filepath)#get file path
            list_name_image_file.append(filename)
            
    for root, dirname, file in os.walk(folder_image_annotation): 
        for filename in file:
            filepath=os.path.join(root,filename)
            list_annotation_file.append(filepath)#get file path
   
    img_map=np.zeros((256,256));
    tmp_folder="tmp";
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder);
        
    annotation_file="test\\annotations\\VOC2007_106.mat"
    file="test\\images\\VOC2007_106.jpg"
    TestPhase("map_img.txt", list_image_file,list_annotation_file )
# #         img_map=createImageHist(annotation_fixed,img,img_map,file, tmp_folder);
# #         boxes, result=Detect_HSV(img, lower,higher);
#         
# 

def Detect_HSV(img, lowerb, upperb):
    img_hsv=cv2.cvtColor(img, cv.CV_BGR2HSV)
    skinMask=cv2.inRange(img_hsv, lowerb, upperb)

    img_hsv=skinMask
    skin_boxes=[]
    #using an elliptical kernel
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    skinMask=cv2.erode(skinMask, kernel, iterations=2)
    skinMask=cv2.dilate(skinMask, kernel, iterations=2)
    # blur the mask to help remove noise, then apply the
    # mask to img
    skinMask=cv2.GaussianBlur(skinMask, (3,3), 0)
    skin_hsv=cv2.bitwise_and(img, img, mask=skinMask)
   
    #detect contour
    contours, hierarchy = cv2.findContours(img_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 100:
            cv2.drawContours(img, contours, i, (0, 255, 0), 3)
            rect_box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(rect_box)
            box = np.int0(box)
#             cv2.drawContours(skin_hsv,[box],0,(0,0,255),2)
            #calculate bounding box
            w=int(np.max([box[0][0],box[1][0],box[2][0],box[3][0]])-np.min([box[0][0],box[1][0],box[2][0],box[3][0]]))
            h=int(np.max([box[0][1],box[1][1],box[2][1],box[3][1]])-np.min([box[0][1],box[1][1],box[2][1],box[3][1]]))
            x_start=np.min([box[0][0],box[1][0],box[2][0],box[3][0]])
            y_start=np.min([box[0][1],box[1][1],box[2][1],box[3][1]])
            skin_boxes.append([int(x_start), int(y_start), w, h])

    return skin_boxes, img
'''
####################################################
## Write the accuracy rate of the test set
####################################################
'''
def writeInformationAccuracy(result_file,name, TP, NP, N, theta):
    #open file
    file=open(result_file,"w");
    file.write("File Name \t\t\t\t\t" +"\tTP \t"+"NP \t"+"N \t"+"  ACCURACY_RATE \n" );
    Acc_Rate=0;
    Sum_Rate=0;
    for name_file,TP_i,NP_i,N_i in zip(name,TP, NP, N):
        if TP_i>N_i:
            TP_i=N_i;
        Acc_Rate=TP_i/float(N_i);
        file.write(name_file +"\t\t\t"+str(TP_i)+" \t"+str(NP_i)+" \t "+str(N_i)+" \t\t\t"+str(Acc_Rate)+"\n");
        Sum_Rate=Sum_Rate+Acc_Rate;
    file.write("\nAverage Accuraccy Rate: "+str(Sum_Rate/len(N)));
    file.write("\nThresold: "+str(theta));
    #close file
    file.close();
    return Sum_Rate/len(N);

def box_overlap(A, B):
    # (x1,y1) top-left coord, (x2,y2) bottom-right coord, (w,h) size
    SA=A[:,2]*A[:,3];
    SB=B[:,2]*B[:,3];
    A_x2=A[:,0]+A[:,2];
    A_y2=A[:,1]+A[:,3];
    B_x2=B[:,0]+B[:,2];
    B_y2=B[:,1]+B[:,3];
    S_intersect=np.max([0, (np.min([A_x2, B_x2])-np.max([A[:,0],B[:,0]])+1)])*np.max([0,(np.min([A_y2,B_y2])-np.max([A[:,1], B[:,1]])+1)])
    S_Union=SA+SB-S_intersect;
    o=float(S_intersect)/float(S_Union);
    print 'overlap between A and B: %f' % o
    
def imageMap(img_map): 
#     hist_img=np.ndarray((img_map.shape[0],img_map.shape[1],3),np.uint8)
    hist_img=np.ndarray((img_map.shape[0],img_map.shape[1]),np.float32)
    min_value=max_value=idx_min=idx_max=0
    min_value, max_value, idx_min, idx_max=cv2.minMaxLoc(img_map)
    print min_value, max_value, idx_min, idx_max
    #Do something here
    intensity=0;
    
    for hbin in xrange(img_map.shape[0]):
        for vbin in xrange(img_map.shape[1]):
            binVal = img_map[hbin,vbin]
#             intensity = cv.Round(binVal*255/max_value)
            hist_img[hbin, vbin]=binVal;

    return hist_img

def writemap(filename, img_map):
    #open file
    file=open(filename,'w');
    file.write("\n");
    for ubin in xrange(img_map.shape[0]):
        for vbin in xrange(img_map.shape[1]):
            file.write(str(vbin) +"\t"+ str(ubin)+"\t"+str(img_map[vbin,ubin])+"\n");
    file.close();
    
def createImageHist(annotation_fixed, img, img_map, name, dir_stored):
    print "Image Hist";
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    idx=1
    s_name=name.split("_");
    s_name=s_name[len(s_name)-1].split(".");
    for (x,y,w,h) in annotation_fixed:
        if y<0:
            y=0;
        if x<0:
            x=0;
        iimg_hsv=img_hsv[y:y+h,x:x+w];
       
        for i in xrange(iimg_hsv.shape[0]):
            for j in xrange(iimg_hsv.shape[1]):
                img_map[iimg_hsv[i,j][0],iimg_hsv[i,j][1]]=img_map[iimg_hsv[i,j][0], iimg_hsv[i,j][1]]+1;
        #create name file
        name_stored=dir_stored+"\\" +s_name[0]+"_"+str(idx)+".bmp";  
        print name_stored;
        img_stored=imageMap(img_map);
        cv2.imwrite(name_stored, img_stored);
        idx=idx+1
        
    return img_map;

'''
#####################################################
##Read mat file of the annotation of the hand images
#####################################################
'''
def loadMatFile(file_name, img):
    data_contents=sio.loadmat(file_name);
    annotation= data_contents['boxes'];
#     drawing=np.zeros((450,450,3),dtype=np.uint8);
    print '+++++++++++++++++++++++++';
    c, r=annotation.shape[:2];
    annotation_fixed=[]
    for i in range(r):
        x1=annotation[0,i]['a'][0,0]
        pt1=x1[0]
        x2=annotation[0,i]['b'][0,0]
        pt2=x2[0]
        x3=annotation[0,i]['c'][0,0]
        pt3=x3[0]
        x4=annotation[0,i]['d'][0,0]
        pt4=x4[0]

        w=int(np.max([pt1[1],pt2[1],pt3[1],pt4[1]])-np.min([pt1[1],pt2[1],pt3[1],pt4[1]]))
        h=int(np.max([pt1[0],pt2[0],pt3[0],pt4[0]])-np.min([pt1[0],pt2[0],pt3[0],pt4[0]]))
        x_start=np.min([pt1[1],pt2[1],pt3[1],pt4[1]])
        y_start=np.min([pt1[0],pt2[0],pt3[0],pt4[0]])
        annotation_fixed.append([int(x_start), int(y_start), w, h])

    return annotation,annotation_fixed, img;
'''
#####################################
##calculate Threshold
#####################################
'''
def checkexist(hsv_value, value):
    idx=-1;
    for x in xrange(len(hsv_value)):
        if hsv_value[x][0]==value[0]:
            if hsv_value[x][1]==value[1]:
                if hsv_value[x][2]==value[2]:
                    return True, idx;
    return False, idx;

def calHSVHist(img, hsv_value, hsv_freq):
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            exist_, idx=checkexist(hsv_value, img[x,y])
            if exist_==True:
               hsv_freq[idx]=hsv_freq[idx]+1;
               continue;
            else:
               hsv_value.append(img[x,y]);
               hsv_freq.append(1);
               
    for x in xrange(len(hsv_value)):
        print hsv_value[x];#,hsv_freq[x];
    
    return hsv_value, hsv_freq;

def writeInformationHist(result_file, hist, hsv_freq):
    #open file
    file=open(result_file,"w");
    file.write("\n" );
    for x in xrange(len(hist)):
       print hsv_freq[x]
       file.write(str(hist[x][0])+","+str(hist[x][1])+","+str(hist[x][2])+" \t " +str(hsv_freq[x])+"\n");
   
    #close file
    file.close();
    
def getThreshHSV(annotation_fixed, img,hist_hsv, hist_freq):
    
    lower_value=-1.0;
    high_value=-1.0;
    lower_value_hsv=[];
    high_value_hsv=[];
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    
    for (x,y,w,h) in annotation_fixed:
        if y<0:
            y=0;
        if x<0:
            x=0;
            
        iimg_hsv=img_hsv[y:y+h,x:x+w];
        iimg_gray=img[y:y+h,x:x+w];
        iimg_bgr=img[y:y+h,x:x+w];
        hist_hsv, hist_freq=calHSVHist(iimg_hsv,hist_hsv, hist_freq);
        writeInformationHist("hist.txt", hist_hsv, hist_freq)

        #plt.hist(hist_hsv.ravel(),256,[0,256])
#         plt.hist(img_hist,256,[0,256])
#         plt.title('Histogram for gray scale picture')
#         plt.show()
      
    return hist_hsv, hist_freq;

def drawImage(img, boxes, annotation_fixed):
#     print boxes
#     print annotation_fixed
    for (x,y,w,h) in boxes:
        cv2.rectangle(img, (x,y), (x+w,y+h),255,2)
    for (x,y,w,h) in annotation_fixed:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255),1)
    cv2.imshow('result 1', img)
    cv2.waitKey(0)

def calculateAvegrate(input_folder):
    list_image_file=[]
    list_name_image_file=[]
    threshold_data=[]
    for root, dirname, file in os.walk(input_folder): 
        for filename in file:
            filepath=os.path.join(root,filename)
            list_image_file.append(filepath)#get file path
            list_name_image_file.append(filename)
    
    #create list file
    img_avg=np.zeros((256,256),np.float32);
    count=0;

    for file in list_image_file:
        #Step1: read image
        print file;
        img=cv2.imread(file,0)
        #Step 2: calculate threshold
        local_threshold=np.sum(img/255.0);

        threshold_data.append(local_threshold)
        img_avg=img_avg+img;
        count=count+1;

    img_avg=(img_avg/count);
    print "-----------------" 
    
    cv2.imwrite('avg.bmp', img_avg); 
    threshold_data.sort()
    np.savetxt('map_img.txt', threshold_data, fmt='%-5.2f')

def Test(name):
    I=cv2.imread(name,0);
    cv2.imshow('winname', I)
    cv2.waitKey(0);
    min_value, max_value, idx_min, idx_max=cv2.minMaxLoc(I)
    print min_value, max_value, idx_min, idx_max
#     np.savetxt('map_img.txt',I, fmt='%-5.2f')
    file=open('map_img.txt','w');
    for hbin in xrange(I.shape[0]):
        for vbin in xrange(I.shape[1]):
#             if I[hbin, vbin]==max_value:
            file.write(str(hbin)+"\t"+str(vbin)+" "+str(I[hbin, vbin])+"\n");
    file.close()
    
def ScaleThreshold(filename):
    I=cv2.imread(filename,0);
    higher_threshold=[];
    lower_threshold=[];
    b=False;
    for hbin in xrange(I.shape[0]):
        if b==True:
            break;
        for vbin in xrange(I.shape[1]):
            if I[(I.shape[0]-hbin-1)][ (I.shape[1]-vbin-1)]!=0:
                higher_threshold.append([(I.shape[0]-hbin-1), (I.shape[1]-vbin-1), I[(I.shape[0]-hbin-1)][ (I.shape[1]-vbin-1)]]);
                b=True;
                break;
    
    b=False;
    for hbin in xrange(I.shape[0]):
        if b==True:
            break;
        for vbin in xrange(I.shape[1]):
            if I[hbin][vbin]!=0:
                lower_threshold.append([hbin,vbin, I[hbin][vbin]]);
                b=True;
                break;
    return lower_threshold, higher_threshold  

def readArray(filename):
    theta_list=[];
    f=open(filename,'r')
    for line in f:
        theta_list.append(float(line))
    return theta_list

def TestPhase(file_thresold, list_image_file,list_annotation_file ):
    #calculate threshold value;
    theta=readArray("map_img.txt")
    size_threshold=100;
#     theta=np.sum(M1/255.0);
    boxes=[]
    TP=[];
    NP=[];
    N=[];
    
    for idx in xrange(size_threshold,1,-2):
        
    #create list file
        for file, annotation_file in zip(list_image_file,list_annotation_file) :
#         #Step1: read image
#           print file;
#             annotation_file="test\\annotations\\VOC2007_106.mat"
#             file="test\\images\\VOC2007_106.jpg"
            img=cv2.imread(file)
    #         #Step 2: detect bounding boxes
    #         boxes=np.asarray(boxes)
    #         print "elapsed times: ", time.time()-t0
    #       #Step 3: read information from annountation mat file
            
            annotation, annotation_fixed, img=loadMatFile(annotation_file, img);
            window_detected=TestImage("avg.bmp",file, annotation_fixed, theta[idx]);
        
            #calculate Accuracy rate
            window_detected=np.asarray(window_detected)
            TP_i=NP_i=0;    
            N_i=len(window_detected);
        
            if len(window_detected)>0:
                TP_i, NP_i, N_i=boxoverlap(window_detected,annotation_fixed,0.1);
                TP.append(TP_i);
                NP.append(NP_i);
                N.append(N_i);
                
        ###############################################################    
        #Show result 
        #Write Accuracy Rate to file
        ###############################################################
        if len(TP)>0:
            ACC=writeInformationAccuracy('result_overlap.txt', list_image_file, TP, NP, N, theta[idx]); 
            print ACC;
            if ACC>=0.7:  
                break;              
        print len(TP), len(NP)
        
        if len(TP)>0:
            del TP[:];  
            del NP[:]; 
        
        print "*************************"
        print theta[idx]
            
def TestImage(filename_map, filename, annotationbox, theta):
    
    img=cv2.imread(filename);
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_canny=cv2.Canny(gray, 100, 200)
    M=cv2.imread(filename_map,0);
    
    #convert HSV
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    
    size_w=int(hsv.shape[0]/7);
    size_h=int(hsv.shape[1]/7);
    
    print size_w, size_h, hsv.shape[0], hsv.shape[1]
    
    window_detected=[];
    #create map image for test image
    for i in xrange(0,hsv.shape[0],int(size_w)):
        if (i+size_w)>hsv.shape[0]:
            break;
        for j in xrange(0,hsv.shape[1],int(size_h)):
           
            if (j+size_h)>hsv.shape[1]:
                break
          
            window_hsv=hsv[i:i+size_w,j:j+size_h];

            if len(window_hsv)==0:
                print (j+size_h), hsv.shape[1]
                break;
#             cv2.imshow('HSV', window_hsv)
#             cv2.waitKey(0)
            
            I=np.zeros((256,256),np.float32);
           
            for x in xrange(window_hsv.shape[0]):
                for y in xrange(window_hsv.shape[1]):
                    I[window_hsv[x,y][0],window_hsv[x,y][1]]=I[window_hsv[x,y][0], window_hsv[x,y][1]]+1;
                    
            #calculate K image
            K=(M/255.0)*I;
            #calculate threshold          
            v=np.sum(K);
                
            if v>theta:
                window_detected.append([j,i,size_h,size_w])
                
    print "----------"
    #draw window detected and annotation  
#     drawImage(img, window_detected, annotationbox);         
    return window_detected
    
'''
##############################################
##check box overlap
##############################################
'''
def boxoverlap(regions_a, region_b, thre):
    # (x1,y1) top-left coord, (x2,y2) bottom-right coord, (w,h) size
    TP=NP=0;
    TP_all=NP_all=0
    N=len(region_b);
    
    for (xb,yb,wb,hb) in region_b:
        x1=np.maximum(regions_a[:,0],xb);
        y1=np.maximum(regions_a[:,1],yb);
        x2=np.minimum((regions_a[:,2]+regions_a[:,0]),(xb+wb));
        y2=np.minimum((regions_a[:,3]+regions_a[:,1]),(yb+hb));
        print x1,y1,x2,y2
        w=x2-x1+1;
        h=y2-y1+1;
        inter=w*h;
        aarea=(regions_a[:,2]+1)*(regions_a[:,3]+1);
        barea=(wb+1)*(hb+1);

        #intersection over union overlap
        o=inter/(aarea+float(barea)-inter);
        
        #set invalid entries to 0 overlap
        o[w<=0]=0
        o[h<=0]=0
        TP=len(np.extract(o>=thre, o))
        NP=len(np.extract(o<thre, o))
        TP_all=TP_all+TP
        
    NP_all=NP-TP_all
    if NP_all<0:
        NP_all=0
        
    return TP_all, NP_all, N; 
  
                   
if __name__ == '__main__':
    name='test\\';
    name1='D:\\Database\\Database SL\\hand_dataset\\hand_dataset\\test_dataset\\test_data'
#     lower, higher=ScaleThreshold('avg.bmp');
    detectHandFolder(name1);
#     calculateAvegrate('tmp\\')
#     Test('avg.bmp');
