import os
import cv2
import numpy as np
from sklearn.metrics import mean_absolute_error


def obrada_videa(video_putanja):
    broj_udaraca = 0

    broj_frejma = 0
    cap = cv2.VideoCapture(video_putanja)
    cap.set(1, broj_frejma)  # indeksiranje frejmova
    ret_val, prvi_frejm = cap.read()

    p_vertikale = racunanje_vertikala(prvi_frejm)
    p_leva_strana = [p_vertikale[15], p_vertikale[16], p_vertikale[17]]
    p_leva_unutrs = [p_vertikale[35], p_vertikale[36], p_vertikale[37]]
    p_desna_strana = [p_vertikale[379], p_vertikale[380], p_vertikale[381], p_vertikale[382]]
    p_desna_unutrs = [p_vertikale[359], p_vertikale[360], p_vertikale[361], p_vertikale[362]]

    while True:
        broj_frejma += 1
        povratna, frejm = cap.read()

        # ako frejm nije zahvacen
        if not povratna:
            break

        trenutne_vertikale = racunanje_vertikala(frejm)

        t_leva_strana = [trenutne_vertikale[15], trenutne_vertikale[16], trenutne_vertikale[17]]
        t_leva_unutrs = [trenutne_vertikale[35], trenutne_vertikale[36], trenutne_vertikale[37]]
        t_desna_strana = [trenutne_vertikale[379], trenutne_vertikale[380], trenutne_vertikale[381], trenutne_vertikale[382]]
        t_desna_unutrs = [trenutne_vertikale[359], trenutne_vertikale[360], trenutne_vertikale[361], trenutne_vertikale[362]]

        if(p_leva_strana != t_leva_strana and p_leva_unutrs == t_leva_unutrs): broj_udaraca += 1
        if(p_desna_strana != t_desna_strana and p_desna_unutrs == t_desna_unutrs): broj_udaraca += 1

        p_vertikale = trenutne_vertikale
        p_leva_strana = [p_vertikale[15], p_vertikale[16], p_vertikale[17]]
        p_leva_unutrs = [p_vertikale[35], p_vertikale[36], p_vertikale[37]]
        p_desna_strana = [p_vertikale[379], p_vertikale[380], p_vertikale[381], p_vertikale[382]]
        p_desna_unutrs = [p_vertikale[359], p_vertikale[360], p_vertikale[361], p_vertikale[362]]

    return broj_udaraca

def racunanje_vertikala(frejm):
    isecak = frejm[105:585, 280:680]
    visina, sirina, kanali = isecak.shape

    gray = cv2.cvtColor(isecak, cv2.COLOR_BGR2GRAY)

    ret, gray_threshed = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    bilateral_filtered_image = cv2.bilateralFilter(gray_threshed, 5, 175, 175)

    edges = cv2.Canny(bilateral_filtered_image, 230, 250)

    indeksi = np.where(edges != [0])
    coordinates = zip(indeksi[0], indeksi[1])

    vertikale = {}  # racunam maks. visinu na svakoj x koordinati na osnovu slike koju je izbacio canny detektor
    for par in coordinates:
        if par[0] >= 405 and par[0] <= 427:
            continue

        if par[1] not in vertikale:
            vertikale[par[1]] = 1
        else:
            vertikale[par[1]] += 1

    for x in range(sirina):
        if x not in vertikale: vertikale[x] = 0

    return vertikale

tacne_vrednosti = {}
dobijene_vrednosti = {}

tacne_lista = []

f = open("dataset/res.txt", "r")
for linija in f:
    if linija.split(',')[0] == 'file': continue
    tacne_vrednosti[linija.split(',')[0]] = eval(linija.split(',')[1].strip())

for klip in sorted(tacne_vrednosti):
    tacne_lista.append(tacne_vrednosti[klip])

for file in os.listdir("dataset"):
    if file.endswith(".mp4"):
        dobijena = obrada_videa(os.path.join("dataset", file))
        print(file, dobijena)
        dobijene_vrednosti[file] = dobijena


dobijene_lista = list(dobijene_vrednosti.values())

print("MAE: ",mean_absolute_error(tacne_lista, dobijene_lista))