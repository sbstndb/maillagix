import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from multiprocessing import Pool, cpu_count
import argparse

# Paramètres globaux
num_frames = 26  # Nombre total de frames
grid_size_x = 2048  # Taille totale de la grille en x
grid_size_y = 2048  # Taille totale de la grille en y
num_ranks_x = 2  # Nombre de rangs en x
num_ranks_y = 2  # Nombre de rangs en y
subgrid_size_x = grid_size_x // num_ranks_x  # Taille de la sous-grille en x par rang
subgrid_size_y = grid_size_y // num_ranks_y  # Taille de la sous-grille en y par rang

# Fonction pour charger une sous-frame d'un rang spécifique
def load_subframe(args):
    frame_idx, i, j = args
    filename = f'frame_{frame_idx:03d}_{i}_{j}.txt'  # Ex: frame_000_0_0.txt
    if os.path.exists(filename):
        subframe = np.loadtxt(filename)
        return (i, j, subframe)
    else:
        return (i, j, None)

# Fonction pour générer et enregistrer une frame
def generate_frame(args):
    frame_idx, output_dir = args
    full_frame = np.zeros((grid_size_y, grid_size_x))
    tasks = [(frame_idx, i, j) for i in range(num_ranks_x) for j in range(num_ranks_y)]
    
    # Charger les sous-frames séquentiellement pour ce frame
    subframes = [load_subframe(task) for task in tasks]
    
    # Assembler la frame
    for i, j, subframe in subframes:
        if subframe is not None:
            start_x = i * subgrid_size_x
            end_x = start_x + subgrid_size_x
            start_y = j * subgrid_size_y
            end_y = start_y + subgrid_size_y
            full_frame[start_y:end_y, start_x:end_x] = subframe
    
    # Enregistrer la frame si elle contient des données
    if np.any(full_frame):
        output_file = os.path.join(output_dir, f"frame_{frame_idx:03d}.npy")
        np.save(output_file, full_frame)
        print(f"Frame {frame_idx} enregistrée dans {output_file}")
    else:
        print(f"Frame {frame_idx} vide, non enregistrée")

# Fonction pour générer toutes les frames
def generate_all_frames(output_dir):
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Nombre de travailleurs
    num_workers = min(4, cpu_count())
    
    # Générer les frames en parallèle
    tasks = [(i, output_dir) for i in range(num_frames)]
    with Pool(num_workers) as pool:
        pool.map(generate_frame, tasks)

# Fonction pour créer l'animation à partir des frames
def create_animation(input_dir, output_gif):
    # Charger les frames depuis le dossier
    frames = []
    for frame_idx in range(num_frames):
        frame_file = os.path.join(input_dir, f"frame_{frame_idx:03d}.npy")
        if os.path.exists(frame_file):
            frame = np.load(frame_file)
            frames.append(frame)
        else:
            print(f"Frame {frame_idx} manquante ({frame_file} non trouvé)")

    # Vérifier s'il y a des frames à afficher
    if not frames:
        print("Aucun frame trouvé dans le dossier. Arrêt du script.")
        return

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
    ani.save(output_gif, writer='pillow', fps=10)
    print(f"Animation enregistrée dans {output_gif}")

# Gestion des arguments en ligne de commande
def main():
    parser = argparse.ArgumentParser(description="Générer des frames ou une animation à partir de sous-frames MPI.")
    parser.add_argument("mode", choices=["generate", "animate"], 
                        help="Mode: 'generate' pour créer les frames, 'animate' pour générer l'animation")
    parser.add_argument("--folder", type=str, default="frames", 
                        help="Nom du dossier pour enregistrer/lire les frames (défaut: 'frames')")
    parser.add_argument("--output", type=str, default="animation.gif", 
                        help="Nom du fichier GIF de sortie (défaut: 'animation.gif', utilisé uniquement en mode 'animate')")
    
    args = parser.parse_args()

    if args.mode == "generate":
        print(f"Génération des frames dans le dossier '{args.folder}'...")
        generate_all_frames(args.folder)
    elif args.mode == "animate":
        print(f"Création de l'animation à partir du dossier '{args.folder}'...")
        create_animation(args.folder, args.output)

if __name__ == "__main__":
    main()
