from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
#from shapely.geometry import LineString
import os
import math
import pandas as pd


def principal_axis(img):
  #cv2_imshow(img)
  h,w=img.shape[:2]
  d=[]
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  for j in range(w):
    f=0
    for k in range(h):
      if img[k,j]== 255:
        f=1
    if f==0:
      d.append(j)
  for v in d[::-1]:
    img=np.delete(img,v,axis=1)

  thresh = cv2.threshold(img, 100 , 255, cv2.THRESH_BINARY)[1]
  
  # find largest contour
  contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]
  big_contour = max(contours, key=cv2.contourArea)

  # fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree 
  ellipse = cv2.fitEllipse(big_contour)
  (xc,yc),(d1,d2),angle = ellipse
  #print(xc,yc,d1,d1,angle)
  rmajor = 1000
  if angle > 90:
      angle = angle - 90
  else:
      angle = angle + 90
  #print(angle)
  xtop = xc + math.cos(math.radians(angle))*rmajor
  ytop = yc + math.sin(math.radians(angle))*rmajor
  xbot = xc + math.cos(math.radians(angle+180))*rmajor
  ybot = yc + math.sin(math.radians(angle+180))*rmajor
  #print(xtop,ytop,xbot,ybot)
  #cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 3)
  
  return (xtop,ytop),(xbot,ybot)

def drawLine(image,p1,p2):
  theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
  endpt_x1 = int(p2[0] - 200*np.cos(theta))
  endpt_y1= int(p2[1] - 200*np.sin(theta))
  endpt_x2 = int(p2[0] + 200*np.cos(theta))
  endpt_y2= int(p2[1] + 200*np.sin(theta))
  
  #cv2.line(image, (endpt_x1, endpt_y1), (endpt_x2, endpt_y2), 255, 1)
  #cv2_imshow(image)
  rv,cv=line(endpt_x1, endpt_y1, endpt_x2, endpt_y2)
  b=list(zip(rv,cv))
  return b
fol="/content/drive/MyDrive/comparative_studies/IAM/female_writing"
folder=os.listdir(fol)[:750]

#print(folder)
for pa in folder :
  path=os.path.join(fol,pa)
  img=cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU) 
  img=thresh1
  print(path)
  endpoint1,endpoint2=principal_axis(img)   
  
  w=img.shape[1]
  h=img.shape[0]
  e1x=min(w-1,int(endpoint1[0]))
  e1y=min(h-1,int(endpoint1[1]))
  e2x=min(int(endpoint2[0]),w-1)
  e2y=min(int(endpoint2[1]),h-1)
  e1x=max(0,e1x)
  e1y=max(0,e1y)
  e2x=max(0,e2x)
  e2y=max(0,e2y)

  mask = np.zeros((h,w), np.uint8)
  pts = np.array([[0,0],[0,e2y],[w-1,e1y],[w-1,0]])
  _=cv2.drawContours(mask, np.int32([pts]),0, 255, -1)
  
  ## Step 4: do mask-op
  img1 = img.copy()
  img2 = img.copy()
  img1[mask==0] = 0
  img2[mask>0] = 0
  cropped=[img1,img2]
  
  
  #print(e1x,e1y,e2x,e2y)
  rr,cc=line(e1x,e1y,e2x,e2y)
  a=list(zip(rr,cc))
  #print(a)
  im = Image.open(path)
  im=im.convert('1')
  px = im.load()
  w,h=im.size
  points=[]
  alen=len(a)
  upperpoint=[]
  lowerpoint=[]
  bu=[]
  feature_vector1=[]
  temp=[]
  temp2=[]
  feature_vector2=[]
  for j in range(alen):
    x2=a[j][0]
    y2=a[j][1]
    #print(x2,y2)
    x1=e1x
    y1=e1y
    xdif = x2 - x1
    ydif = y2 - y1
    a1 = x2 - ydif // 2
    b1 = y2 + xdif // 2
    a2 = x2 + ydif // 2
    b2 = y2 - xdif // 2
    st=drawLine(img,(a1,b1),(x2,y2))
    i=0
    l=len(st)
    f=0
    p=-1
    q=-1
    #print(px[200,0])
    while i<l:
      u=int(st[i][0])
      v=int(st[i][1])
      if u<0 or u>=w or v<0 or v>=h:
        i=i+1
      elif px[u,v]>0:
        p=(u,v)
        i=l
      else:
        i=i+1
    i=l-1
    while i>=0:
      u=int(st[i][0])
      v=int(st[i][1])
      if u<0 or u>=w or v<0 or v>=h:
        i=i-1
      elif px[u,v]>0:
        q=(u,v)
        i=-1
        
      else:
        i=i-1
    if p==-1 or q==-1:
      continue
    else:
      upperpoint.append(p)
      lowerpoint.append(q)
      if p!=q:
        res1=(p[0]+q[0])//2
        res2=(p[1]+q[1])//2
        points.append((res1,res2))
        d=math.sqrt( ((res1-x2)**2)+((res2-y2)**2) )
        temp.append(d)
  l=len(points)
  temp=temp[:1200]
  temp=[0]+[v for v in temp]+[0 for j in range(1200-l)]
  print(l)
  temp3=[0]
  '''temp4=[1]
  for j in range(l-1):
    d=math.sqrt( ((points[j][0]-points[j+1][0])**2)+((points[j][1]-points[j+1][1])**2))
    temp4.append(d)'''

  for img in cropped:
    path=img
    endpoint1,endpoint2=principal_axis(path)   
    
    w=img.shape[1]
    h=img.shape[0]
    e1x=min(w-1,int(endpoint1[0]))
    e1y=min(h-1,int(endpoint1[1]))
    e2x=min(int(endpoint2[0]),w-1)
    e2y=min(int(endpoint2[1]),h-1)
    e1x=max(0,e1x)
    e1y=max(0,e1y)
    e2x=max(0,e2x)
    e2y=max(0,e2y)
    rr,cc=line(e1x,e1y,e2x,e2y)
    a=list(zip(rr,cc))
    #print(a)
    im =  Image.fromarray(img)
    im=im.convert('1')
    px = im.load()
    w,h=im.size
    points=[]
    alen=len(a)
    upperpoint=[]
    lowerpoint=[]
    bu=[]
    feature_vector1=[]
    temp2=[]
    feature_vector2=[]
    for j in range(alen):
      x2=a[j][0]
      y2=a[j][1]
      #print(x2,y2)
      x1=e1x
      y1=e1y
      xdif = x2 - x1
      ydif = y2 - y1
      a1 = x2 - ydif // 2
      b1 = y2 + xdif // 2
      a2 = x2 + ydif // 2
      b2 = y2 - xdif // 2
      st=drawLine(img,(a1,b1),(x2,y2))
      i=0
      l=len(st)
      f=0
      p=-1
      q=-1
      #print(px[200,0])
      while i<l:
        u=int(st[i][0])
        v=int(st[i][1])
        if u<0 or u>=w or v<0 or v>=h:
          i=i+1
        elif px[u,v]>0:
          p=(u,v)
          i=l
        else:
          i=i+1
      i=l-1
      while i>=0:
        u=int(st[i][0])
        v=int(st[i][1])
        if u<0 or u>=w or v<0 or v>=h:
          i=i-1
        elif px[u,v]>0:
          q=(u,v)
          i=-1
          
        else:
          i=i-1
      if p==-1 or q==-1:
        continue
      else:
        upperpoint.append(p)
        lowerpoint.append(q)
        if p!=q:
          res1=(p[0]+q[0])//2
          res2=(p[1]+q[1])//2
          points.append((res1,res2))
          d=math.sqrt( ((res1-x2)**2)+((res2-y2)**2) )
          temp2.append(d)
    l=len(points)
    '''temp4=[]
    for j in range(l-1):
      d=math.sqrt( ((points[j][0]-points[j+1][0])**2)+((points[j][1]-points[j+1][1])**2))
      temp4.append(d)'''

    #print(temp2[:10])
    temp2=temp2[:1200]
    temp=temp[:]+[v for v in temp2]+[0 for j in range(1200-l)]
    print(l)
    
  df = pd.DataFrame([temp])
  df.to_csv('/content/experiment2IAM.csv',mode='a',index=False,header=False)
  #print(len(points))

  

  
  

