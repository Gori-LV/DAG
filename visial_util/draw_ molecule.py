# import rdkit
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import pandas as pd
# from rdkit.Chem import Draw

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem import rdBase,Draw,AllChem
from rdkit.Chem import Draw
# from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
# import networkx as nx
# from networkx.readwrite import cytoscape_data
# import cyjupyter
# from cyjupyter import Cytoscape
from rdkit.Chem.Scaffolds import rdScaffoldNetwork
# from urllib import parse
#
# N#Cc1ccc3c(c1)Cc2cc([N+](=O)O)ccc23
# O=[N+](O)c3cc[c+]2c1CCCCc1nc2c3
# C[N+]1=NC2C(=C1)C=CC=C2[N+](=O)O
with open('MUTAG smiles/mutag_188_data.can','r') as f:
    content = f.readlines()
smiles = [x.strip().split(' ')[0] for x in content]
#
# for i in range(len(smiles)):
#     mo = smiles[i]
#     try:
#         # Draw.MolToFile(Chem.MolFromSmiles(mo),'MUTAG/'+str(i)+'.svg',imageType='svg')
#         Draw.MolToFile(Chem.MolFromSmiles(mo),'MUTAG/'+str(i)+'.png')
#     except:
#         print(i) #82 187

# non_mtg = [39, 41, 97, 110, 142, 154]
non_mtg = [39, 97, 110, 142, 154]
to_draw = [Chem.MolFromSmiles(smiles[mo]) for mo in non_mtg]
img=Draw.MolsToGridImage(to_draw, molsPerRow=6,useSVG=True)
with open('non_mtg.svg','w') as f:
    f.write(img)

# mtg = [25, 101, 104, 165, 176, 179]
mtg = [25, 104, 165, 176, 179]
to_draw = [Chem.MolFromSmiles(smiles[mo]) for mo in mtg+non_mtg]
# img=Draw.MolsToGridImage(to_draw, molsPerRow=6,useSVG=True)
img=Draw.MolsToGridImage(to_draw,molsPerRow=5,subImgSize=(200,200),useSVG=True)

with open('5e.svg','w') as f:
    f.write(img)


exp1 = [25, 49]

exp2 = [104, 169, 175, 177]
to_draw = [Chem.MolFromSmiles(smiles[mo]) for mo in exp1]+ [Chem.MolFromSmiles('O=[N+](O)c3ccc2c(ccc1ccccc12)c3')]+[Chem.MolFromSmiles(smiles[mo]) for mo in exp2]
# img=Draw.MolsToGridImage(to_draw, molsPerRow=6,useSVG=True)
# img=Draw.MolsToGridImage(to_draw,molsPerRow=5,subImgSize=(200,200),useSVG=True)
img=Draw.MolsToGridImage(to_draw,molsPerRow=7,subImgSize=(200,200),useSVG=True)
with open('exp.svg','w') as f:
    f.write(img)

img.save('test.svg')
for i in non_mtg:
    mo = smiles[i]
    try:
        Draw.MolToFile(Chem.MolFromSmiles(mo),'MUTAG/'+str(i)+'.svg',imageType='svg')
        # Draw.MolToFile(Chem.MolFromSmiles(mo),'MUTAG/'+str(i)+'.png')
    except:
        print(i) #82 187
