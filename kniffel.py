import random

import numpy as np


def würfeln(n):
    würfel = []
    for i in range(n):
        wurf = random.randint(1, 6)
        würfel.append(wurf)
    return würfel

def benutzer_fragen(n):
    zahlen = []
    zahl = input("Möchtest du eine Zahl behalten?")
    if zahl == "Nein" or zahl == "nein":
        return zahlen
    zahlen.append(zahl)
    for i in range(n - 1):
        zahl = input("Möchtest du noch eine Zahl behalten?")
        if zahl == "Nein" or zahl == "nein":
            return zahlen
        else:
            zahlen.append(zahl)
    return zahlen


def main():
    n = 5
    würfel = würfeln(n)
    print(würfel)
    zahlen_behalten_1 = benutzer_fragen(n)
    print("Rausgelegt: " + str(zahlen_behalten_1))
    würfel = würfeln(n - len(zahlen_behalten_1))
    print(würfel)
    zahlen_behalten_2 = benutzer_fragen(n)
    zahlen_behalten_1.extend(zahlen_behalten_2)
    print("Rausgelegt: " + str(zahlen_behalten_1))
    würfel = würfeln(n - len(zahlen_behalten_1))
    print(würfel)



if __name__ == "__main__":
    main()
