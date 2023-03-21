import os

import pygame
import numpy as np
import imageio

import cv2


def pca(X, samples=10):

    # flatten the images
    X = X.reshape(X.shape[0], -1)

    mean = X.mean(axis=0)  # get the mean
    Xp = (X - mean) # normalize the data

    # get the covariance matrix
    cov = np.cov(Xp, rowvar=False)

    # get the eigenvalues and eigenvectors
    E, U = np.linalg.eig(cov)  # E is the eigenvalues, U is the eigenvectors

    idx = np.argsort(E)[::-1]  # sorting the eigenvalues

    U_length = U.shape[1]
    U = U[:, idx]  # sorting the eigenvectors

    reconstructions = []
    for i in range(1, samples + 1):
        U_copy = U[:, :int(U_length * (i / samples))]

        projection = np.dot(U_copy.T, Xp)

        reconstructions.append(np.dot(U_copy,projection).T + mean) # image reconstruction

    return reconstructions


def main():
    # load images from olivetti_faces.npy
    data = np.load("olivetti_faces.npy")[56]

    reconstructions = pca(data, 20)  # selected eigenvectors (64) from the 4096 eigenvectors

    # save first 10 images
    for i in range(len(reconstructions)):
        reconstructions[i] *= 255  # or any coefficient
        reconstructions[i] = reconstructions[i].astype(np.uint8)
        reconstructions[i] = cv2.cvtColor(reconstructions[i], cv2.COLOR_GRAY2RGB)
        imageio.imwrite("reconstruction_" + str(i) + ".png", reconstructions[i])


    pygame.init()
    screen = pygame.display.set_mode([x*2 for x in data.shape])
    pygame.display.set_caption("Hello World")
    clock = pygame.time.Clock()
    running = True
    i = len(reconstructions) - 1
    background = np.zeros((*[x*2 for x in data.shape], 3), dtype=np.uint8)
    while running:
        # blit on screen the images from the reconstruction list in the middle of the screen (640, 480)
        # use the mouse wheel to scroll through the images
        background[:data.shape[0] * 2, :data.shape[1] * 2, :] = cv2.resize(reconstructions[i], (data.shape[1] * 2, data.shape[0] * 2))
        screen.blit(pygame.surfarray.make_surface(background), (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    if i < len(reconstructions) - 1:
                        i += 1
                        print(i)
                elif event.y < 0:
                    if i > 0:
                        i -= 1
                        print(i)
        clock.tick(60)
        pygame.display.flip()


if __name__ == "__main__":
    main()
