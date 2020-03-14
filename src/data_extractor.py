import os
import shutil
from os.path import dirname, realpath
os.chdir(realpath(dirname(__file__)))

from copy import copy
from functools import reduce
import numpy as np
import pandas as pd

def prettify_csv():

	"""
	Gemmer dataene i noegletal.npy, hvor hver fils navn angiver det pågældende nøgletal
	På 0.-aksen er kommunen
	På 1.-aksen er årstallet (hvor indeks 0 svarer til 1993 og 26 til 2019)
	På 2.-aksen er det pågældende nøgletal
	I attrs.txt gemmes på første linje alle kommunerne, på anden linje årstal og på tredje nøgletal
	"""
	
	with open("full_noegletal.csv", encoding="utf-8") as nt:
		dat = np.empty((98, 27, 0))
		# Liste med nøgletal
		noegletal = []
		# Liste med kommuner
		kommuner = []
		has_kommuner = False
		# Liste med årstal
		aarstal = [str(x) for x in range(1993, 2020)]
		for i, line in enumerate(nt):
			# Splitter linjen op
			entries = line.strip().split(";")
			# De første fem linjer indeholder ikke data
			if i < 4: continue
			# Ved 'Tegnforklaring' stopper dataene
			if line.startswith("Tegnforklaring"): break

			# Datalinje
			if all(entries):
				if not has_kommuner:
					kommuner.append(entries[0].split()[0].lower())
				data_entries = []
				for entry in entries[2:]:
					try:
						entry = entry.replace(".","")
						data_entries.append(float(entry.replace(",", ".")))
					except ValueError:
						data_entries.append(None)
				dat[j, :, -1] = data_entries
				j += 1

			# Ved kun én indgang starter en ny kategori
			if entries[0] and not any(entries[1:]):
				noegletal.append(entries[0].lower())
				dat = np.concatenate((dat, np.empty((98, 27, 1))), axis=2)
				j = 0  # Tæller antal linjer i en given kategori
				continue
			
			# Ved indholdsløs linje slutter en kategori, og filen gemmes
			if not any(entries):
				# Hvis hele den netop fundne kategori er nans, fjernes den
				if np.isnan(dat[:, :, -1]).all():
					dat = dat[:, :, :-1]
					noegletal.pop(-1)
				has_kommuner = True
				continue
	
	# Gemmer en version uden nans
	dat_min = dat.copy()
	noegletal_min = copy(noegletal)
	for i in range(len(noegletal)-1, 0, -1):
		if np.isnan(dat[:, :, i]).any():
			dat_min = np.concatenate((dat_min[:, :, :i], dat_min[:, :, i+1:]), axis=2)
			noegletal_min.pop(i)

	print("Fuldt datasæt:")
	print("Gemte %i værdier for %i nøgletal" % (reduce(lambda x, y: x * y, dat.shape), dat.shape[-1]))
	nans = np.count_nonzero(np.isnan(dat))
	print("Heraf %i manglende værdier (%.2f %%)" % (nans, nans/dat.size*100))

	print("Datasæt uden nans:")
	print("Gemte %i værdier for %i nøgletal" % (reduce(lambda x, y: x * y, dat_min.shape), dat_min.shape[-1]))
	nans_min = np.count_nonzero(np.isnan(dat_min))
	print("Heraf %i manglende værdier (%.2f %%)" % (nans_min, nans_min/dat_min.size*100))

	np.save("noegletal", dat)
	np.save("noegletal_min", dat_min)
	with open("attrs.out", "w", encoding="utf-8") as a, open("attrs_min.out", "w", encoding="utf-8") as a_min:
		a.write("\n".join([";".join(kommuner), ";".join(aarstal), ";".join(noegletal)]))
		a_min.write("\n".join([";".join(kommuner), ";".join(aarstal), ";".join(noegletal_min)]))


def get_data(kommuner="all", aarstal="all", noegletal="all", use_min=False):
    """
    Returnerer en m x n x o-matrix, hvor m er antal kommuner, n antal aarstal og o antal noegletal
    Parametrene skal enten være all, én værdi eller arraylike med de ønskede værdier
    """

    kommuner = "all" if kommuner == "all" else [x.lower() for x in kommuner]
    aarstal = "all" if aarstal == "all" else [x.lower() for x in aarstal]
    noegletal = "all" if noegletal == "all" else [x.lower() for x in noegletal]

    with open("attrs%s.out" % ("_min" if use_min else ""), encoding="utf-8") as p:
        attrs = p.read().split("\n")
        attrs = {
                "kommuner": attrs[0].split(";"),
                "aarstal": attrs[1].split(";"),
                "noegletal": attrs[2].split(";"),
                }

    if kommuner == "all":
      kommuner = attrs["kommuner"]
    if aarstal == "all":
      aarstal = attrs["aarstal"]
    if noegletal == "all":
      noegletal = attrs["noegletal"]
    kommune_indices = [
            attrs["kommuner"].index(x) for x in kommuner
            ]
    aarstal_indices = [
            attrs["aarstal"].index(x) for x in aarstal
            ]
    noegletal_indices = [
            attrs["noegletal"].index(x) for x in noegletal
            ]

    data = np.load("noegletal%s.npy" % ("_min" if use_min else ""))
    data = data[kommune_indices, :, :][:, aarstal_indices, :][:, :, noegletal_indices]
    return data

"""
Gemmer alle noegletal som csv-filer i en bestemt mappe med en bestemt delimiter
"""
def save_all_to_csv(folder="default_csv", deli=";"):

    #Lav folder til at gemme csv filer
    path = os.path.join(os.getcwd(), folder)
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)

    #Hent np array over alle noegletal
    data = get_data(use_min=False)

    #Hent navne paa kommuner
    with open("attrs.out", encoding="utf-8") as p:
        attrs = p.read().split("\n")
        attrs = {
			"kommuner": attrs[0].split(";"),
			"aarstal": attrs[1].split(";"),
			"noegletal": attrs[2].split(";"),
		}


    for idx, d in enumerate(data):
        df = pd.DataFrame(d, index=attrs["aarstal"], columns=attrs["noegletal"])
        df.to_csv(os.path.join(path, attrs["kommuner"][idx]+".csv"), sep=deli)


if __name__ == "__main__":

	prettify_csv()
	save_all_to_csv()



