import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from multiprocessing import Pool, cpu_count

# Paramètres
num_frames = 26  # Nombre total de frames à charger
grid_size = 1024  # Taille de la grille
dx = 1.0 / grid_size
dy = 1.0 / grid_size

# Fonction pour charger une frame
def load_frame(i):
    filename = f'frame_{i:03d}.txt'  # Nom du fichier, e.g., frame_000.txt
    if os.path.exists(filename):
        return np.loadtxt(filename)  # Charge le fichier s'il existe
    else:
        return None  # Retourne None si le fichier est manquant

# Nombre de travailleurs (8 par défaut, mais limité par le nombre de cœurs)
num_workers = min(8, cpu_count())

# Charger les frames en parallèle
with Pool(num_workers) as pool:
    frames = pool.map(load_frame, range(num_frames))

# Filtrer les frames valides (supprimer les None)
frames = [frame for frame in frames if frame is not None]

# Vérifier s'il y a des frames à afficher
if not frames:
    print("Aucun frame trouvé. Arrêt du script.")
    exit()

# Configurer la figure pour l'animation
fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Simulation d\'advection')
ax.set_aspect('equal')

# Afficher la première frame
im = ax.imshow(frames[0],
               extent=[0, 1, 0, 1],  # Domaine de 0 à 1
               origin='lower',       # Origine en bas à gauche
               cmap='jet',           # Palette de couleurs
               vmin=0, vmax=1)       # Échelle de couleurs
fig.colorbar(im)  # Ajouter une barre de couleur

# Fonction de mise à jour pour l'animation
def update(frame):
    im.set_array(frame)
    return [im]

# Créer l'animation
ani = animation.FuncAnimation(fig, update, frames=frames,
                              interval=100,  # 100 ms par frame
                              blit=True)     # Optimisation

# Sauvegarder en GIF
ani.save('animation.gif', writer='pillow', fps=10)

# Optionnel : afficher l'animation
# plt.show()