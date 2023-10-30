import os
import shutil

def copy_mp4_files(src_folder, dest_folder):

    # Créez le dossier de destination s'il n'existe pas
    os.makedirs(dest_folder, exist_ok=True)

    # Vérifiez si le dossier 'triche' existe dans le dossier source donné
    for run in os.listdir(src_folder):
        if not run.startswith('run'):
            continue

        triche_folder = os.path.join(src_folder, run + '/triche')
        if not os.path.exists(triche_folder):
            print(f'Aucun dossier "triche" trouvé dans {triche_folder}')
            continue
        elif len(os.listdir(triche_folder)) is 0:
            print(f'Aucune vidéo trouvée dans {triche_folder}')
            continue
        
        if not os.path.exists(triche_folder[0:-6] + 'org'):
            print(f'Aucun dossier "org" trouvé dans {triche_folder}')
            continue

        origin_list = os.listdir(triche_folder[0:-6] + 'org')
        for filename in os.listdir(triche_folder):
            if filename.endswith('.mp4'):
                # Construisez les chemins complets du fichier source et du fichier de destination
                id = origin_list.index(filename[0:18] + 'org' + filename[21:])
                src_file_path  = os.path.join(triche_folder[0:-6] + 'org', origin_list[id-1])
                dest_file_path = os.path.join(dest_folder, filename[0:18] + 'arg' + filename[21:])
                shutil.copy(src_file_path, dest_file_path)

                src_file_path  = os.path.join(triche_folder[0:-6] + 'org', origin_list[id])
                dest_file_path = os.path.join(dest_folder, filename[0:18] + 'org' + filename[21:])
                shutil.copy(src_file_path, dest_file_path)
                
                print(f'Copié {triche_folder, filename} vers {dest_folder}')

# Spécifiez les dossiers source et destination

machine_list= ['00007_btl_new/demo_0927', '00321-D', '00322-G', '00323-M']

src_folder  = './' + machine_list[1]
dest_folder = '../../ia_ob2_training/dataset_20231005/video'

# Appelez la fonction
for i in range(len(machine_list)):
    src_folder  = './' + machine_list[i]
    copy_mp4_files(src_folder, dest_folder)