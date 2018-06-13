import numpy as np
import PIL
from functools import reduce
from scipy.stats import linregress
from itertools import chain
from math import log2
from itertools import product as cartesian
from matplotlib import pyplot as plt
from itertools import permutations
from random import shuffle


quarter = lambda M: chain.from_iterable(
                    np.split(m,2,1) for m in np.split(M,2,0))

vsum =lambda *Xs: list(map(sum,zip(*Xs)))

def dbox(I):
    def get_N(I,N):
        N.append(I.sum())
        n=len(I)
        if n==1: return N
        else:
            J=np.zeros((n//2,n//2),dtype=int)
            for i in range(n):
                for j in range(n):
                    J[i//2,j//2]+=I[i,j]
            return get_N(J.clip(0,1),N)
                         
    N=list( reversed( get_N(I, [])))
    #print( N);print(111)
    return linregress(range(len(N)), list(map(log2,N)))[0]

def dbox2(I):
    def get_N(I):
        if I.size==1: return I.A1
        else:
            N = sum([get_N(i) for i in quarter(I)])
            #print( N, N[0])
            return np.append(N,min(1 , N[-1]))
    
    N=list(reversed(get_N(I)))
    #print( N);print(111)
    return linregress(range(len(N)), list(map(log2,N)))[0]


zoom_in = lambda x, scale: scale*(x % (1./scale))

def invert(A,t, mode='<='):
    if mode=='<=': return (A <= t).astype(int)
    else: return (A >= t).astype(int)

matrix_range=lambda m,n: map(np.array, cartesian(range(m),range(n)))
indices=lambda shape: cartesian(range(shape[0]),range(shape[1]))

pad0 = lambda a,n:np.pad(a,(0,n),mode='constant',constant_values=0)

def make_fractal_image(S, n, stopshort=False):
    
    img=S
    listS=S.tolist()
    values = set.union(*(set(row) for row in listS))
    
    if stopshort:
        m = n//len(S)
    else: m = n
    
    while len(img) <= m:
        d= dict((v,v*img) for v in values)
        new=[]
        for row in listS:
            new.append([])
            for entry in row:
                new[-1].append(d[entry])
        img = np.bmat(new)

    plt.imshow(img[:n,:n],interpolation='nearest',cmap='hot')
    plt.show()
    if len(img)>=n:
        return img[:n,:n]
    else:
        return np.matrix(pad0(img,n-len(img)))
		
    
    if len(img) == n: return img
    else:
        plt.imshow(img[:n,:n],interpolation='nearest',cmap='hot')
        plt.show()
        input("jjdjdjd")
        return np.asmatrix(PIL.Image.fromarray(img).resize((n,n)))

def permute(A, perm=None):
    L=A.A1.tolist()
    shuffle(L)
    return np.matrix(L).reshape(A.shape)
    
    
##    if perm is None:
##        from random import shuffle
##        perm = list(indices(A.shape))
##        shuffle(perm)
##        
##    new = np.matrix(np.zeros(A.shape))
##    indic=indices(A.shape)
##    for index in perm:
##        new[index]=A[indic.__next__()]
##    return new

avgFS4=np.matrix((1.160964047443573, 1.5849625007212829, 1.850219859070538))

def fractal_excess(img, dim=None):
    if dim is None: dim=dbox(img)
    return dbox(permute(img))-dim

blockofones=lambda n,shape: np.matrix(
    np.array((1,)*n+(0,)*(shape[0]*shape[1]-n)).reshape(shape))
    
mxsum = lambda A: sum(sum(row) for row in A.tolist())

def fracmax(m=2):
    from itertools import combinations
    avg = lambda X: sum(X)/len(X)
    d={}
    for i in range(1,m+1):
        n=2**i
        d[n]=[[0]]
        print(n)
        indic=list(indices((n,n)))
        A=np.matrix(np.zeros((n,n)))
        for k in range((n**2)):
            
            A[k//n, k%n]=1
            d[n].append([])
            for C in combinations(indic, k+1):
                B=permute(A, C)
                d[n][-1].append(dbox(B))
            
            print(k+1, avg(d[n][-1]),max(d[n][-1]))
        print()
    return d

def nPr(n,r):
    ret=1
    for i in range(n-r+1,n+1):
        ret*=i
    return ret
            
def nCr(n,r):
    if n-r < r: r=n-r
    return nPr(n,r)/nPr(r,r)
            
def avgfrac(n, m):
    '''n=2**k is size of square matrix; m<=n is # of pixels "on"'''
    from itertools import combinations
    avg = lambda X: sum(X)/len(X)
    L=[0]*2**n
    S=0

    for C in combinations(range(n**2),m):
        #print(C)
        for index in C:
            L[index]=1

        A = np.matrix(L).reshape((n,n))
        S += dbox(A)        

        for index in C:
            L[index]=0

    return S/nCr(n**2,m)

def sortedmatrix(A):
    return np.matrix(sorted(sum(A.tolist(),[]))).reshape(A.shape)

    
#to do:        
from math import ceil
intceil=lambda x: int(ceil(x))

Image=PIL.Image
Image.Image.pltshow= (
    lambda self, **args:
    plt.show( plt.imshow( self,
                          **dict([['cmap','gray']]
                                 +
                                 list(args.items()))
                          )
              )
    )
Image.Image.getmatrix= (
    lambda self:
    np.matrix(list(self.getdata())) .reshape(self.size)
    )

#img.pltshow()
#print( img.format,img.size,img.mode)
import sys
#print('sys.argv',sys.argv)

def dothing(infile):
    original=Image.open(infile)
    
    size=(2**int( log2( min(original.size))), )*2
    img=original.resize(size)
    original.close()

    img=img.convert("L")
    mat=img.getmatrix()
    #img.close()

    sigma = mat.std()
    if sigma==0:
        raise ValueError("This image is featureless and dull")
    
    lower = mat.mean()-1.5*sigma
    upper = lower + 3*sigma
    lower, upper = np.array((lower,upper)).clip(0,255)

    T = int(lower)+1
    L = upper - T
    while L <= 6:
        print("Widening T range...")
        lower, upper = np.array((lower-1,upper+1)).clip(0,255)
        T=int(lower)+1
        L = upper - T

    S = list(range(T, int(T+L) if L == int(L) else int(T+L+1)))


    def getFSFE(A,P=None,mode='<='):
        if P is None: P = permute(A)
        FS = []
        FE = []
        
        for T in S:
            FS.append(dbox(invert(A, T,mode)))
            FE.append(dbox(invert(P, T,mode)) - FS[-1])

        return FS,FE

    P = permute(mat)
    
    from time import time
    
    print("count black")
    t=time()
    black=getFSFE(mat,P)
    print( time()-t)
    print('\n')
    with open(infile+"--black.log",'w') as f:
        f.write(str(black))
    

    print("count white")
    t=time()
    white=getFSFE( mat, P,'>=')
    print(time()-t)
    with open(infile+"--white.log",'w') as f:
        f.write(str(white))
    
 
    return black,white


    #return img

#img=dothing("bonobo.jpg")

if len(sys.argv)>1:
    blacks=[]
    whites=[]
    for infile in sys.argv[1:]:
        print(infile)
        black,white=dothing(infile)
        print('\n')
        print(infile)
        blacks.append(black)
        whites.append(white)

    plt.plot(np.linspace(-1.5,1.5,len(blacks[0][1])),blacks[0][1],
             np.linspace(-1.5,1.5,len(blacks[1][1])),blacks[1][1],'--')
    plt.figure()
    plt.plot(np.linspace(-1.5,1.5,len(whites[0][1])),whites[0][1],
             np.linspace(-1.5,1.5,len(whites[1][1])),whites[1][1],'--')

            
            
            
            
            
    
#print( shuffled(np.matrix('1 2; 3 4; 5 6; 7 8')))
#seed=lambda X: not( all( X < 2/3) and all(1/3 < X))

#S = make_fractal(seed,3)
##S=np.matrix('1 1;1 0')
##
##n=512
##img=make_fractal_image(S,n)
##from time import time
##t=time()
##print(dbox(img))#,fractal_excess(img))
##print(time()-t)
##print()
##t=time()
##print(dbox2(img))#,fractal_excess(img))
##print(time()-t)
##


#print("min dbox?",
#    dbox(blockofones(mxsum(img),img.shape)))

#img=make_fractal_image(S,n,True)
#print("stopshort",fractal_excess(img))    
    
#print( img)
#plt.imshow(blockofones(mxsum(img),img.shape),interpolation='nearest',cmap='hot')
#plt.show()

##def extend_fractal(seed, depth, scale, old=None):
##    if old is None:
##        old = seed
##        depth -= 1
##                          
##    if depth==0: return old
##    else:
##        return extend_fractal(
##            seed,
##            depth-1,
##            scale,
##            lambda x: (seed(x) and old(zoom_in(x, scale))))
##            
##def make_fractal(seed,scale):
##    '''returns a boolean function G:(a,L)->{True,False} such that
##    G(a,L) is True iff the *open* interval (a,a+L) (or box with corner a, side
##    length L) contains a point of the fractal'''
##
##    from math import floor, ceil,log
##    from random import shuffle
##    ifloor = lambda x: int(floor(x))
##    iceil = lambda x: int(ceil(x))
##    
##    def G(a,L1,L2=None, eps=2**-20):
##    
##        k = iceil( -log(L1/10.)/log(scale))
##        fractal = extend_fractal(seed, k, scale)
##
##        E = pow(scale,-k)
##        e=np.array((E/2.,E/2.))
##
##        m = ifloor(a[0]/E)
##        n = iceil((a[0]+L1)/E)-1
##        p = ifloor(a[1]/E)
##        q = iceil((a[1]+L1)/E)-1
##        
##
##        points = [E*(np.array((m,p))+X)+e for X in matrix_range(n-m+1,q-p+1)]
##        shuffle(points)
##        for X in points:
##            if fractal(X): return True
##
##        return False
##    return G
##
##        def boundary(E,m,n,p,q):
##            m,n,p,q = (E*val for val in (m,n,p,q))
##            e = E/2
##            #X = np.array(m+e
##
##        def interior(E,m,n,p,q):
##            pass
##
##        if L2 is None:
##            boxes = ((a,L1,L1),)
##        else:
##            boxes = ((a,L1,L2),)
##
##        while boxes != ()
##            for box in boxes:
##                if interior(box): return True
##            newboxes = ()
##            for box in boxes:
##                newboxes += boundary(box)
##            boxes = newboxes
##
##        return false
##
##        while True:
##            
##
##            if interior(): return True
##            elif E < eps: return False
##            elif boundary():
##                k+= 1
##            else: return False


##def make_fractal_image(S, n):
##    img = np.matrix(np.zeros((n,n)))
##    L = 1/n
##    for i,j in matrix_range(n,n):
##        a = L*np.array((j,n-1-i))
##        img[i,j] = int(S(a,L))
##    return img

