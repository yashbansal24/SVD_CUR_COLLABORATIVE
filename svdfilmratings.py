import numpy as np
from scipy.stats import mstats
import scipy
from numpy import linalg
from numpy.linalg import matrix_rank
from math import *
import pandas as pd
import pandas
import time
from math import sqrt
# get max user id and item id

"""
Variables are initialized and libraries are imported here.

"""
max_userid = 1000
max_movieid = 1000

test_set = 200

col_sq_sum = np.zeros(1005)
row_sq_sum = np.zeros(1005)
non_zero_row = np.zeros(1005)
non_zero_col = np.zeros(1005)

"""

initialize function takes input the original matrix and calculate non zero rows and columns

"""
def initialize(list_c, rows, cols):

    for k in range(0, rows):
        for l in range(0, cols):
            if(list_c[k][l]!=0):
                non_zero_col[l] +=1
                non_zero_row[k] +=1


"""
normalize_ratings takes input the matrix and normalize based
on the rating
"""

def normalize_ratings(list_c, rows, cols):

    col_sum = np.zeros(1005)
    row_sum = np.zeros(1005)

    avg_user_rat = np.zeros(1005)
    global col_sq_sum
    global row_sq_sum
    for k in range(0, rows):
        for l in range(0, cols):
            col_sum[l]+=(list_c[k][l])
            col_sq_sum +=(list_c[k][l]**2)
            row_sum[k]+=(list_c[k][l])
            row_sq_sum +=(list_c[k][l]**2)


    for k in range(0, rows):
        if(non_zero_row[k] != 0):
            avg_user_rat[k] += row_sum[k]/non_zero_row[k]

    for k in range(0, rows):
        for l in range(0, cols):
            if list_c[k][l] != 0:
                list_c[k][l] -= avg_user_rat[k]


    return list_c


"""

normalizations functions is used to normalize U and V matrix for SVD.
This takes input a matrix and divide the whole column by the column sum.

"""
def normalizations(list_c, rows, cols):
    for k in range(0, cols):
        sum_c = 0
        for l in range(0, rows):
            sum_c += (list_c[l][k] ** 2)
        su_sqr = abs(sqrt(sum_c))
        for l in range(0, rows):
            if su_sqr != 0:
                list_c[l][k] = (list_c[l][k]/su_sqr)
    return list_c


"""

This part of the code is scanning the input from the file ratings.txt

"""
start_time = time.time()
userid = []
itemid = []
rating = []
size_of_ratings = 0
with open("ratings.txt", "r") as f:
    data = f.read().split("\n")
    for line in data:
        sd = line.strip(" ").split(" ")
        if line == '':
            continue
        #if(int(sd[0])<=max_userid):

        #if(int(sd[1])<=max_movieid):

        if(int(sd[0])<=max_userid and int(sd[1])<=max_movieid):
            userid.append(int(sd[0]))
            itemid.append(int(sd[1]))
            rating.append(float(sd[2]))
            size_of_ratings+=1



"""
This part of the code transfers the read input to dat_matrix
dat_matrix is copied to A_matrix which is later
used for all the calculations

"""


# Add data to the n*m matrix 0-based indexing NOTE: 0 not to be considered

dat_mat = []
for i in range(0, max_userid):
    l_ch = []
    for j in range(0, max_movieid):
        l_ch.append(0)
    dat_mat.append(l_ch)

# add item to data

for i in range(0, size_of_ratings):
    dat_mat[userid[i]-1][itemid[i]-1] = float(rating[i])

initialize(dat_mat,max_userid,max_movieid)
A_matrix = np.array(dat_mat)
# generating Rank of a matrix
rank = matrix_rank(A_matrix)

# In[23]:

"""
This part of the code is calculation of U matrix
"""

# Generating Left Singular Vectors
A_matrix_transpose = A_matrix.transpose()

A_Atrans = A_matrix.dot(A_matrix_transpose)
print(A_Atrans.shape)
eigenvalues = linalg.eig(A_Atrans)
# print(eigenvalues[0])
# here the first row of the list gives the eigen values of LAMBDA and the second eigen vectors
eigen_values = eigenvalues[0]
eigen_vectors = eigenvalues[1].real
r_eigen_values = []  # get all real positive eigen values

"""
This part of the code is calculation of eigen_values and eigen_vectors
Later they are used for U matrix
"""

for i in eigen_values:
    if i.imag == 0:
        if i.real > 0.00000000:
            r_eigen_values.append(i.real)

U = []
for i in range(0, max_userid):
    U.append(eigen_vectors[i][0:rank])
u_rows = len(U)
u_cols = len(U[0])

normalized_U = normalizations(U, u_rows, u_cols)
############ normalized U matrix calculation

sorted(r_eigen_values, reverse=True)
size_eigens = len(r_eigen_values)
sigma = np.zeros(shape=(rank, rank))

"""
This part of the code is calculation of sigma matrix
sigma matrix is a diagonal matrix

"""

for i in range(0, rank):
    sigma[i][i] = sqrt(abs(eigen_values[i]))
print(sigma.shape)
# .......

eigen_vectors = eigenvalues[1].real
# print(sigma)

# generating V transpose right singular vector At *A
v_mat = np.dot(A_matrix_transpose, A_matrix)
print(v_mat.shape)
v_rows = v_mat.shape[0]
v_eigens = linalg.eig(v_mat)
v_eigen_size = len(v_eigens[1][0])

# print((v_eigens[1]))
"""
This part of the code is calculation of eigen_values and eigen_vectors for V transpose matrix
Later they are used for U matrix
"""
V = []
for i in range(0, v_rows):
    l_reals = v_eigens[1][i].real
    V.append(l_reals[0:rank])
v_transpose = np.array(V).transpose()
v_transpose_l = list(v_transpose)
v_transpose_size = v_transpose.shape
print(v_transpose_size)
v_rows = v_transpose_size[0]
v_cols = v_transpose_size[1]

normalized_V = normalizations(v_transpose_l, v_rows, v_cols)
############ normalized V matrix calculation



normalized_V_arr = np.array(normalized_V)
print(normalized_V_arr.shape)



"""
This part of the code is just for deleting
 the smallest value of sigma reducing the dimension from all three matrix.
 This part is just for testing purposes. This matrix is not used later.

"""


normalized_U_deleted = []
for row in normalized_U:
    normalized_U_deleted.append(row[0:u_cols-1])
print(str(len(normalized_U_deleted)) + "," + str(len(normalized_U_deleted[0])))
# print(normalized_U_deleted)


# Delete last row and colum of sigma
sigma_deleted = []
for i in range(0, u_cols - 1):
    sigma_deleted.append(sigma[i][0:(u_cols-1)])
print(str(len(sigma_deleted)) + "," + str(len(sigma_deleted[0])))
# print(sigma_deleted)


# Delete Last row of V_transpose
v_transpose_deleted = []
for i in range(0, (v_rows-1)):
    v_transpose_deleted.append(v_transpose_l[i])
print(str(len(v_transpose_deleted)) + "," + str(len(v_transpose_deleted[0])))
# print(v_transpose_deleted)
#  Final Check

"""
This is the matrix multiplication of U, sigma and V transpose matrix_rank
"""
check = np.array(normalized_U).dot(np.array(sigma)).dot(np.array(v_transpose_l))
check_sh = check.shape
# print(check)
#print(linalg.norm(abs(A_matrix - check)))

print


"""
calculation of the metrices
spearman
Root mean square error
precision on top 100 = 1 - rmse on top 100
time taken by program

"""

## CALCULATING RMSE AND Spearman
prediction = (check[max_userid-test_set:max_userid,:].ravel() - A_matrix[max_userid-test_set:max_userid,:].ravel())[0]
d2 = (np.dot(prediction,prediction)).sum()
nn2 = non_zero_row[max_userid-test_set:].sum()
n=nn2
#print d2,nn2
nn2 = nn2*(nn2*nn2 - 1)
spear = 1.0 - (6.000000000000000000000000000*d2/(1.*nn2))
print "spearman rank coeff for svd : " + str(spear)

prediction = check[max_userid-test_set:max_userid,:] - A_matrix[max_userid-test_set:max_userid,:]
print "RMSE for svd : " + str(linalg.norm(abs(prediction))/(1.0*n))
prediction = check[max_userid-test_set:max_userid-100,:] - A_matrix[max_userid-test_set:max_userid-100,:]
print "Precisin on top 100 for svd : " + str(1 - linalg.norm(abs(prediction))/(1.0*100*max_movieid))
end_time = time.time() - start_time
print("Time taken to execute the SVD algorithm " + str(end_time))



print


"""

Calculation of the 90% energy threshold.
This part find the dimension at which we can retain 90% energy.

"""



################# 90% svd_retain_energy
start_time = time.time()
s = np.diag(sigma)
diag_sum = float(np.dot(s,s))
s = s.tolist()
#print s
temp=0
for i in range(0,len(s)):
    if(temp>.9*diag_sum):
        break
    temp+=float(s[i]*s[i])

sigma = np.diag(s[:i])
U = np.asarray(normalized_U)
V= np.asarray(v_transpose_l)
check90 = np.matmul(U[:,:i], sigma)
check90 = np.matmul(check90,V[:i,:])



#print(linalg.norm(abs(A_matrix - check90)))


"""
calculation of the metrices
spearman
Root mean square error
precision on top 100 = 1 - rmse on top 100
time taken by program

"""

## CALCULATING RMSE AND Spearman
prediction = (check90[max_userid-test_set:max_userid,:].ravel() - A_matrix[max_userid-test_set:max_userid,:].ravel())[0]
d2 = (np.dot(prediction,prediction)).sum()
nn2 = non_zero_row[max_userid-test_set:].sum()
n=nn2
#print d2,nn2
nn2 = nn2*(nn2*nn2 - 1)
spear = 1.0 - (6.000000000000000000000000000*d2/(1.*nn2))
print "spearman rank coeff for svd 90 percent : " + str(spear)

prediction = check90[max_userid-test_set:max_userid,:] - A_matrix[max_userid-test_set:max_userid,:]
print "RMSE for svd 90 percent : " + str(linalg.norm(abs(prediction))/(1.0*n))
prediction = check90[max_userid-test_set:max_userid-100,:] - A_matrix[max_userid-test_set:max_userid-100,:]
print "Precisin on top 100 for svd 90 percent : " + str(1 - linalg.norm(abs(prediction))/(1.0*100*max_movieid))


end_time = time.time() - start_time
print("Time taken to execute the SVD 90 percent energy algorithm " + str(end_time))

print



#index = np.arange(1000)
#columns = np.arange(1000)
#X =pd.DataFrame (check,index,columns)
#Y = pd.DataFrame(A_matrix,index,columns)


start_time = time.time()
from scipy import stats
sumc=0

#spearson=stats.spearmanr(A_matrix,check)
#spearson = spearson.tolist()
#print check.tolist()
#for i in range(0,1000):
#        k=(stats.spearmanr(A_matrix[i],check[i]))[0]
#        if(k > -1 and  k < 1):
#            sumc=sumc+ (stats.spearmanr(A_matrix[i],check[i]))[0]
#print sumc
#print A_matrix

"""
This is the part of cur.
Here all the Variables are initialized which
are later used.
"""


copy_A = np.empty_like (A_matrix)
copy_A[:] = A_matrix
cur_matrix = normalize_ratings(A_matrix,max_userid,max_movieid)
#print cur_matrix

#sq_sum_matrix = sum(col_sq_sum)

concepts = 200 #matrix_rank(A_matrix)



"""
This part calculates probabilities of the columns and row.
This is later used for
column select algorithm.
"""

col_sq_sum = col_sq_sum/col_sq_sum.sum()
row_sq_sum = row_sq_sum/row_sq_sum.sum()

elements = np.arange(1,max_userid+1)

row_sq_sum[max_userid-1]=(1-row_sq_sum[:max_userid-1].sum())
col_sq_sum[max_movieid-1]=(1-col_sq_sum[:max_movieid-1].sum())

row_sq_sum = row_sq_sum[:max_userid]
col_sq_sum = col_sq_sum[:max_movieid]

#print row_sq_sum[:max_userid].sum()
#print col_sq_sum[:max_movieid].sum()

rows_no = np.random.choice(elements, concepts , p=row_sq_sum)

cols_no = np.random.choice(elements, concepts , p=col_sq_sum)
W = np.zeros((concepts,concepts))

Columns = np.zeros((concepts,max_userid))
Rows = np.zeros((concepts,max_movieid))

"""
This is where columns select and row select algorithm is performed.
We use the rows_no and col_no array. They are obtained from np.random.choice
function which returns the indexes to be taken
based on their probabilities.
"""

k=0
for i in cols_no:
    temp=cur_matrix[:,i-1]
    Columns[k] = temp/sqrt(concepts*col_sq_sum[i-1])
    k +=1
k=0
for i in rows_no:
    temp=cur_matrix[i-1,:]
    Rows[k] = temp/sqrt(concepts*row_sq_sum[i-1])
    k+=1

"""
CALCULATIion of W matrix.
W[i][j] = cur_matrix[rows_no[i]-1][cols_no[j]-1].
This is the indexing used for W matrix.
"""

for i in range(0,concepts):
    for j in range(0,concepts):
        W[i][j] = cur_matrix[rows_no[i]-1][cols_no[j]-1]

U, s, V = np.linalg.svd(W, full_matrices=True)
s = np.diag(s)

Z = np.dot(s,s)
Z = np.linalg.pinv(Z,rcond=.0001)   #,rcond=.0000001
u = np.matmul(V.T, Z)
u = np.matmul(u,U.T)

cur = np.matmul(Columns.T,u)
cur = np.matmul(cur,Rows)


"""
calculation of the metrices
spearman
Root mean square error
precision on top 100 = 1 - rmse on top 100
time taken by program

"""

## CALCULATING RMSE AND Spearman
prediction = (cur[max_userid-test_set:max_userid,:].ravel() - copy_A[max_userid-test_set:max_userid,:].ravel())[0]
d2 = (np.dot(prediction,prediction)).sum()
nn2 = non_zero_row[test_set:].sum()
n=nn2
#print d2,nn2
nn2 = nn2*(nn2*nn2 - 1)
spear = 1.0 - (6.000000000000000000000000000*d2/(1.*nn2))
print "spearman rank coeff for cur : " + str(spear)

prediction = cur[max_userid-test_set:max_userid,:] - copy_A[max_userid-test_set:max_userid,:]
print "RMSE for cur : " + str(linalg.norm(abs(prediction))/(1.0*n))
prediction = cur[max_userid-test_set:max_userid-100,:] - copy_A[max_userid-test_set:max_userid-100,:]
print "Precisin on top 100 for cur percent : " + str(1 - linalg.norm(abs(prediction))/(1.0*100*max_movieid))
end_time = time.time() - start_time
print("Time taken to execute the cur 90 percent energy algorithm " + str(end_time))
print
############### 90 % energy CUR
start_time = time.time()

diag_sum = float(np.dot(s[0],s[0]))
s = s.tolist()
#print s

"""

Calculation of the 90% energy threshold.
This part find the dimension at which we can retain 90% energy.

"""


temp=0
for i in range(0,len(s)):
    if(temp>.9*diag_sum):
        break
    temp+=float(s[0][i]*s[0][i])

Z = np.diag(s[0][:i])
Z = np.dot(Z,Z)
Z = np.linalg.pinv(Z,rcond=.0001)   #,rcond=.0000001
u = np.matmul(V.T[:,:i], Z)
u = np.matmul(u,U.T[:i,:])
#print

cur = np.matmul(Columns.T,u)
cur = np.matmul(cur,Rows)


"""
calculation of the metrices
spearman
Root mean square error
precision on top 100 = 1 - rmse on top 100
time taken by program

"""

## CALCULATING RMSE AND Spearman
prediction = (cur[max_userid-test_set:max_userid,:].ravel() - copy_A[max_userid-test_set:max_userid,:].ravel())[0]
d2 = (np.dot(prediction,prediction)).sum()
nn2 = non_zero_row[test_set:].sum()
n=nn2
#print d2,nn2
nn2 = nn2*(nn2*nn2 - 1)
spear = 1.0 - (6.000000000000000000000000000*d2/(1.*nn2))
print "spearman rank coeff for cur 90 percent : " + str(spear)

prediction = cur[max_userid-test_set:max_userid,:] - copy_A[max_userid-test_set:max_userid,:]
print "RMSE for cur 90 percent : " + str(linalg.norm(abs(prediction))/(1.0*n))
prediction = cur[max_userid-test_set:max_userid-100,:] - copy_A[max_userid-test_set:max_userid-100,:]
print "Precisin on top 100 for cur 90 percent : " + str(1 - linalg.norm(abs(prediction))/(1.0*100*max_movieid))
end_time = time.time() - start_time
print("Time taken to execute the cur 90 percent energy algorithm " + str(end_time))
print
