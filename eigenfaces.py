import numpy as np
import lib
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimage
import math

####################################################################################################


def load_images(path: str) -> list:
    """
    Load all images in path

    :param path: path of directory containing image files

    :return images: list of images (each image as numpy.ndarray and dtype=float64)
    """
    # 1.1 Laden Sie für jedes Bild in dem Ordner das Bild als numpy.array und
    # speichern Sie es in einer "Datenbank" eine Liste.
    # Tipp: Mit glob.glob("data/train/*") bekommen Sie eine Liste mit allen
    # Dateien in dem angegebenen Verzeichnis.
    images = []

    files = glob.glob(path)
    for one_file in files:
        img = plt.imread(one_file)
        images.append(np.asarray(img, dtype="float64"))
    # 1.2 Geben Sie die Liste zurück
    return images

def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    :param images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    :return D: data matrix that contains the flattened images as rows
    """
    n, m = len(images), images[0].size
    matrix = np.empty((n,m))
    # 2.1 Initalisiere die Datenmatrix mit der richtigen Größe und Typ.
    # Achtung! Welche Dimension hat die Matrix?
    # 2.2 Fügen Sie die Bilder als Zeilen in die Matrix ein.
    for index in range(len(images)):
        matrix[index,:] = images[index].ravel()
        
    # 2.3 Geben Sie die Matrix zurück
    return matrix


def calculate_svd(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform SVD analysis for given data matrix.

    :param D: data matrix of size n x m where n is the number of observations and m the number of variables

    :return eigenvec: matrix containing principal components as rows
    :return singular_values: singular values associated with eigenvectors
    :return mean_data: mean that was subtracted from data
    """
    # 3.1 Berechnen Sie den Mittelpukt der Daten
    # Tipp: Dies ist in einer Zeile möglich (np.mean, besitzt ein Argument names axis)
    mean = D.mean(axis=0)

    # 3.2 Berechnen Sie die Hauptkomponeten sowie die Singulärwerte der ZENTRIERTEN Daten.
    # Dazu können Sie numpy.linalg.svd(..., full_matrices=False) nutzen.
    u, s, vh = np.linalg.svd(D - mean, full_matrices=False)
    return vh, s, mean

    # 3.3 Geben Sie die Hauptkomponenten, die Singulärwerte sowie den Mittelpunkt der Daten zurück


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    :param singular_values: vector containing singular values
    :param threshold: threshold for determining k (default = 0.8)

    :return k: threshold index
    """
    # 4.1 Normalizieren Sie die Singulärwerte d.h. die Summe aller Singlärwerte soll 1 sein

    singular_values = singular_values / singular_values.sum()
    # 4.2 Finden Sie den index k, sodass die ersten k Singulärwerte >= dem Threshold sind.
    k = singular_values.size
    accumulated_energy = 0
    for index, value in enumerate(singular_values):
        accumulated_energy += value
        if accumulated_energy > threshold:
            k = index
            break
    # 4.3 Geben Sie k zurück
    return k

def project_faces(pcs: np.ndarray, mean_data: np.ndarray, images: list) -> np.ndarray:
    """
    Project given image set into basis.

    :param pcs: matrix containing principal components / eigenfunctions as rows
    :param images: original input images from which pcs were created
    :param mean_data: mean data that was subtracted before computation of SVD/PCA

    :return coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """
    # 5.1 Initialisieren Sie die Koeffizienten für die Basis.
    basis = images - mean_data
    # Sie sollen als Zeilen in einem np.array gespeichert werden.

    # 5.1 Berechnen Sie für jedes Bild die Koeffizienten.
    # Achtung! Denkt daran, dass die Daten zentriert werden müssen.
    coefficients = basis @ pcs.T
    # 5.2 Geben Sie die Koeffizenten zurück
    return coefficients


def identify_faces(coeffs_train: np.ndarray, coeffs_test: np.ndarray) -> np.ndarray:
    """
    Perform face recognition for test images assumed to contain faces.
    
    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    :param coeffs_train: coefficients for training images, each image is represented in a row
    :param coeffs_test: coefficients for test images, each image is represented in a row

    :return indices: array containing the indices of the found matches
    """

    indicies = np.zeros(coeffs_test.shape[0], dtype=int)

    for i in range(coeffs_test.shape[0]):
        smallest_angle = math.inf
        for j in range(coeffs_train.shape[0]):
            arc_angle = (coeffs_test[i, :] @ coeffs_train[j, :]) / (np.linalg.norm(coeffs_test[i, :]) * np.linalg.norm(coeffs_train[j, :]))
            current = np.arccos(arc_angle) 
            if current < smallest_angle:
                smallest_angle = current
                indicies[i] = j
    
    return indicies
    # 8.1 Berechnen Sie für jeden Testvektor den nächsten Trainingsvektor.
    # Achtung! Die Distanzfunktion ist definiert über den Winkel zwischen den Vektoren.



if __name__ == '__main__':
    ...
    # 1. Aufgabe: Laden Sie die Trainingsbilder.
    # Implementieren Sie dazu die Funktion load_images.
    train_data = load_images('data/train/*.png')

    # 2. Aufgabe: Konvertieren Sie die Bilder zu Vektoren die Sie alle übereinander speichern,
    # sodass sich eine n x m Matrix ergibt (dabei ist n die Anzahl der Bilder und m die Länge des Bildvektors).
    # Implementieren Sie dazu die Funktion setup_data_matrix.
    train_matrix = setup_data_matrix(train_data)

    # 3. Aufgabe: Finden Sie alle Hauptkomponeten des Datensatztes.
    # Implementieren Sie dazu die Funktion calculate_svd
    component, singular_values, mean = calculate_svd(train_matrix)
    # 4. Aufgabe: Entfernen Sie die "unwichtigsten" Basisvektoren.
    # Implementieren Sie dazu die Funktion accumulated_energy um zu wissen wie viele
    # Baisvektoren behalten werden sollen. Plotten Sie Ihr Ergebniss mittels
    # lib.plot_singular_values_and_energy
    k = accumulated_energy(singular_values)
    lib.plot_singular_values_and_energy(singular_values, k)

    # 5. Aufgabe: Projizieren Sie die Trainingsdaten in den gefundenen k-dimensionalen Raum,
    # indem Sie die Koeffizienten für die gefundene Basis finden.
    # Implementieren Sie dazu die Funktion project_faces
    train_Koeffizienten = project_faces(component[:k, :], mean, train_matrix)
    # 6. Aufgabe: Laden Sie die Test Bilder (load_images).
    test_data = load_images('data/test/*.png')
    test_matrix = setup_data_matrix(test_data)

    # 7. Aufgabe: Projizieren Sie die Testbilder in den gefundenen k-dimensionalen Raum (project_faces).
    test_Koeffizienten = project_faces(component[:k, :], mean, test_matrix)

    # 8. Aufgabe: Berechnen Sie für jedes Testbild das nächste Trainingsbild in dem
    # gefundenen k-dimensionalen Raum. Die Distanzfunktion ist über den Winkel zwischen den Punkten definiert.
    # Implementieren Sie dazu die Funktion identify_faces.
    # Plotten Sie ihr Ergebniss mit der Funktion lib.plot_identified_faces
    matches = identify_faces(train_Koeffizienten, test_Koeffizienten)

    # plot the identified faces
    lib.plot_identified_faces(matches, train_data, test_data, component[:k, :], test_Koeffizienten, mean)