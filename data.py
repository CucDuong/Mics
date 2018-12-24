import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import numpy as np



def resize_all_images(X,size=100):
    X_shape = X.shape;
    num_images = X_shape[3];
    X_new = np.zeros((size,size,3,num_images));
    print ('num_images')
    print (num_images);
    for i in range (num_images):
        X_new[:,:,:,i] = cv2.resize(X[:,:,:,i], (size,size), interpolation = cv2.INTER_CUBIC)
    return X_new

def num2onehot(num):
    out = np.zeros((1,9))
    out[0,num-1] = 1;
    return out

def one_hot_coding(Y):
    Y_one_hot = np.zeros((Y.shape[0],9))
    for i in range (Y.shape[0]):
        Y_one_hot[i,:] = num2onehot(Y[i,0])
    return Y_one_hot

def delete_classes(X,Y):
    index_to_delete=[]
    count=0;
    for i in range (Y.shape[0]):
        if (Y[i,0]==9):
            index_to_delete.append(i)
        if (Y[i,0]==1):
            count +=1
            if (count>10):
                index_to_delete.append(i)
        
    X_new = np.delete(X,index_to_delete,0)
    Y_new = np.delete(Y,index_to_delete,0)
    return X_new, Y_new

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    #nW = int((h * sin) + (w * cos))
    #nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    #M[0, 2] += (nW / 2) - cX
    #M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (w,h))


def get_data(resize = 100):
    X_3_1=sio.loadmat('X_3_1.mat')
    X_3_1_matrix = X_3_1['X_3_1']

    X_3_2=sio.loadmat('X_3_2.mat')
    X_3_2_matrix = X_3_2['X_3_2']

    X_2_1=sio.loadmat('X_2_1.mat')
    X_2_1_matrix = X_2_1['X_2_1']

    X_2_2=sio.loadmat('X_2_2.mat')
    X_2_2_matrix = X_2_2['X_2_2']

    X_120=sio.loadmat('X_120.mat')
    X_120_matrix = X_120['X_120']
    Y_120=sio.loadmat('Y_120.mat')
    Y_120_matrix = Y_120['Y_120']

    X_180=sio.loadmat('X_180.mat')
    X_180_matrix = X_180['X_180']
    Y_180=sio.loadmat('Y_180.mat')
    Y_180_matrix = Y_180['Y_180']

    print ('X_120_matrix')
    print(X_120_matrix.shape)
    print ('Y_120_matrix')
    print(Y_120_matrix.shape)
    print ('X_180_matrix')
    print(X_180_matrix.shape)
    print ('Y_180_matrix')
    print(Y_180_matrix.shape)
    
    target_size = resize;
    
    X_120_matrix_new = resize_all_images(X_120_matrix,size=target_size)
    X_180_matrix_new = resize_all_images(X_180_matrix,size=target_size)
    X_2_1_matrix_new = cv2.resize(X_2_1_matrix, (target_size,target_size), interpolation = cv2.INTER_CUBIC).reshape((target_size,target_size,3,1))
    X_2_2_matrix_new = cv2.resize(X_2_2_matrix, (target_size,target_size), interpolation = cv2.INTER_CUBIC).reshape((target_size,target_size,3,1))
    X_3_1_matrix_new = cv2.resize(X_3_1_matrix, (target_size,target_size), interpolation = cv2.INTER_CUBIC).reshape((target_size,target_size,3,1))
    X_3_2_matrix_new = cv2.resize(X_3_2_matrix, (target_size,target_size), interpolation = cv2.INTER_CUBIC).reshape((target_size,target_size,3,1))



    X_final_matrix = np.concatenate((X_120_matrix_new,X_180_matrix_new,X_2_1_matrix_new,X_2_2_matrix_new,X_3_1_matrix_new,X_3_2_matrix_new), axis=3)
    X_final_matrix_new = np.moveaxis(X_final_matrix,-1,0)
    Y_final_matrix=np.concatenate((Y_120_matrix,Y_180_matrix,(np.array([2,2,3,3])).reshape((4,1))))

    X_new, Y_new = delete_classes(X_final_matrix_new,Y_final_matrix)
    
    X_rotated=[]
    Y_rotated=[]
    for i in range (10,370,40): 
        X_rotated.append(rotate_bound(X_new[-1,:,:,:],10*i))
        Y_rotated.append(Y_new[-1,0])
        X_rotated.append(rotate_bound(X_new[-2,:,:,:],10*i))
        Y_rotated.append(Y_new[-2,0])
        X_rotated.append(rotate_bound(X_new[-3,:,:,:],10*i))
        Y_rotated.append(Y_new[-3,0])
        X_rotated.append(rotate_bound(X_new[-4,:,:,:],10*i))
        Y_rotated.append(Y_new[-4,0])
    
    X_rotated_array=np.array(X_rotated)
    Y_rotated_array=np.array(Y_rotated)
    Y_rotated_array=Y_rotated_array.reshape([Y_rotated_array.shape[0],1])
    X_new = np.concatenate((X_new, X_rotated_array), axis=0)
    Y_new = np.concatenate((Y_new, Y_rotated_array), axis=0)
    plt.hist(Y_new,bins=8)
    return X_new,one_hot_coding(Y_new) ;
