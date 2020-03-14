import numpy as np


numbers = np.load('../noegletal.npy')

def to_int(arr):
    return np.array([int(x) for x in arr])

def year_mask(arr):
    return np.array([x > 2007 and x < 2019 for x in arr])


#Hent navne paa kommuner
with open("../attrs.out", encoding="utf-8") as p:
        attrs = p.read().split("\n")
        attrs = {
            "kommuner": attrs[0].split(";"),
         	"aarstal": attrs[1].split(";"),
         	"noegletal": attrs[2].split(";"),
        }
        idx = attrs['noegletal'].index('anmeldte tyverier/indbrud pr. 1.000 indb.')

        numbers = numbers[:,year_mask(to_int(attrs['aarstal'])),idx]
        burglaries = numbers[:][:][:]
        indexes = np.unravel_index(np.argsort(burglaries, axis=None),np.shape(burglaries))
        burglaries = burglaries[indexes]
        burglaries = (burglaries - np.mean(burglaries))/np.std(burglaries)
        print(min(burglaries))
        print(max(burglaries)/3)
        print(2*max(burglaries)/3)
        print(max(burglaries))
