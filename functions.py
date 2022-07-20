import functions as f
import pickle
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from networkx.algorithms import community
from networkx import edge_betweenness_centrality as betweenness


# Opens a file, returns a dictionary (VARIATION 1)
def open_file(location):
    file = open(location, "r")
    dic = dict()
    for lines in file:
        lines = lines.split("\t")
        dic[(lines[1], lines[2])] = float (lines[3])
    return dic


# opens a file, returns a dictionary (VARIATION 2)
def open_file1(location):
    file = open(location, "r")
    dic = dict()
    for lines in file:
        lines = lines.split("\t")
        dic[(lines[0], lines[1])] = [float(lines[2]),float(lines[3]),float(lines[4]),float(lines[5]),float(lines[6])]
    return dic


# opens a file, returns a dictionary (VARIATION 3)
def open_file2(location):
    file = open(location, "r")
    dic = dict()
    for lines in file:
        lines = lines.replace('\n','').split("\t")
        if len(lines)==1:
            continue
        dic[(lines[0], lines[1])] = [float(lines[2]),float(lines[3]),float(lines[4]),float(lines[5]),float(lines[6])]
    return dic


# creates a file from a dictionary file
def save_file(dictionary, location):
    file = open(location, "w")
    new_dic = dict()
    for key, value in dictionary.items():
        new_dic[key] = value
    for key, value in new_dic.items():
        file.writelines([key[0], "\t", key[1], "\t", str(value[0]), "\t", str(value[1]), "\t", str(value[2]), "\t", str(value[3]), "\t", str(value[4]), "\n"])
    file.close()


# Organizes every line of a file into alphabetical order
# returns the file in list form
def alphabetize(file_location):
    temp_list = []
    file = open(file_location, "r")
    for lines in file:
        temp_list.append(lines)
    temp_list.sort()
    file.close()
    return temp_list


# finds the amount of unique proteins in a dictionary
def find_unique_prots(dic):
    tempList = []
    for key in dic.keys():
        tempList.append(key[0])
        tempList.append(key[1])
    tempSet = set(tempList)
    return tempSet


# finds the amount of unique protein pairs in a dictionary
def find_unique_pairs(dic):
    tempList = []
    for key in dic.keys():
        tempList.append(key)
    tempSet = set(tempList)
    return tempSet


# creates a new file using items from a list
def write_file(file, new_list):
    new_file = open(file, "w")
    for lines in new_list:
        new_file.writelines(lines)
    new_file.close()


# converts proteins into humanprot useable names
def convert_name(dic, gene_dic):
    new_dic = dict()
    for key,value in dic.items():
        if key[0] in gene_dic:
            key = (gene_dic.get(key[0]), key[1])
        if key[1] in gene_dic:
            key = (key[0], gene_dic.get(key[1]))
        new_dic[key] = value
    return new_dic


# takes a list of protein pairs and a list of dictionaries
# creates a dictionary with the protein pairs and their values over each time period
def proper_data(pairList, dictList):
    d = dict()
    for item in pairList:
        d[item] = []
    count = 1
    for location in dictList:
        element = f.open_file(location)
        for key, value in element.items():
            if key in d:
                d[key].append(element[key])
        for key, value in d.items():
            if len(value) != count:
                d[key].append(0)
        count = count + 1

    return d


# removes all keys that have more than 1 zero in their value list
def prune_zeros(location, pruningThreshold):
    file = open(location, "r")
    d = f.open_file2(location)
    tempList = []
    for key, value in d.items():
        count = 0
        for element in value:
            if element == 0:
                count = count + 1
            if count >= pruningThreshold:
                tempList.append(key)
                break

    for element in tempList:
        del d[element]

    return d


# predicts missing data points if the rest of the data follows a linear path
def predict_zeros(dic):
    new_dic = dic
    del_list = []
    for key, value in new_dic.items():
        if 0 in value:
            loc = value.index(0)
            value.remove(0)
            if loc in [0, 2, 4]:
                if abs((value[1] - value[0]) - (value[3] - value[2])) < 0.1:
                    if loc == 0:
                        if ((value[0] - (value[1] - value[0])) < 1) and ((value[0] - (value[1] - value[0])) > 0):
                            value.insert(loc, value[0] - (value[1] - value[0]))
                        else:
                            del_list.append(key)
                    if loc == 2:
                        if ((value[2] - (value[3] - value[2])) < 1) and ((value[2] - (value[3] - value[2])) > 0):
                            value.insert(loc, value[2] - (value[3] - value[2]))
                        else:
                            del_list.append(key)
                    if loc == 4:
                        if ((value[3] + (value[3] - value[2])) < 1) and ((value[3] + (value[3] - value[2])) > 0):
                            value.insert(loc, value[3] + (value[3] - value[2]))
                        else:
                            del_list.append(key)
                else:
                    del_list.append(key)
            elif loc in [1]:
                if abs((value[1] - value[0]) - ((value[3] - value[2]) * 2)) < 0.1:
                    if ((value[1] - (value[3] - value[2])) < 1) and ((value[1] - (value[3] - value[2])) > 0):
                        value.insert(loc, value[1] - (value[3] - value[2]))
                    else:
                        del_list.append(key)
                else:
                    del_list.append(key)
            elif loc in [3]:
                if abs(((value[1] - value[0]) * 2) - (value[3] - value[2])) < 0.1:
                    if ((value[3] - (value[1] - value[0])) < 1) and ((value[3] - (value[1] - value[0])) > 0):
                        value.insert(loc, value[3] - (value[1] - value[0]))
                    else:
                        del_list.append(key)
                else:
                    del_list.append(key)

    for element in del_list:
        del new_dic[element]

    return new_dic


# returns a dictionary of shared proteins
# FORMAT: (prot1, prot2) : [value1, value2...]
# values are the different % chance at different times
def find_shared_prots(dictList):
    d = dict()
    for i, location in enumerate(dictList):
        if i == 0:
            continue
        element = f.open_file(location)
        firstDict = f.open_file(dictList[0])
        for key in firstDict:
            if key in element.keys() or (key[1], key[0]) in element.keys():
                if key in d:
                    d[key].append(element[key])
                else:
                    d[key] = [firstDict[key], element[key]]

    del_list = []
    for key in d.keys():
        if len(d.get(key)) < len(dictList):
            del_list.append(key)

    for element in del_list:
        del d[element]

    return d


# saves obj as pickle file
def save_object(obj, save_name):
    f = open(save_name + ".pkl", "wb")
    pickle.dump(obj, f, -1)
    f.close()


# loads pickle file into obj
def load_object(file_name):
    f = open(file_name + ".pkl", "rb")
    obj = pickle.load(f)
    f.close()
    return obj


# loads a tab file
def load_proteome(address):
    table = pd.read_table(address)
    accesion = list(table['Entry'])
    names = list(table['Gene names'])

    prot_dict = {}
    for i in range(len(accesion)):
        prot_dict[accesion[i]] = names[i]

    return prot_dict


# returns two dictionaries,
# 1) keys are proteins and values are a list of subcellular localizations
# 2) keys are subcellular localization and the values are a set of proteins
def create_subcell_locs():
    def get_subcell_prots(address):
        table = pd.read_table(address, low_memory=False)
        table = table.fillna(value='BADBAD')
        uni_filter = table['Uniprot'] != 'BADBAD'
        table = table[uni_filter]
        evi_filter = table['Evidence'] == 'Evidence at protein level'
        table = table[evi_filter]
        raw_prots = set(table['Uniprot'])
        prots = set()
        for prot in raw_prots:
            prot_list = prot.split(',')
            for p in prot_list:
                prots.add(p)

        return prots

    np_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Nucleoplasm_Nuclear.tsv')
    nuclear_membrane_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Nuclear.tsv')
    nucleoli_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Nucleoli_Nucleoli.tsv')
    actin_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Actin.tsv')
    aggresome_prots = get_subcell_prots(
        './Data/subcellular_locations/subcell_location_Aggresome_Cytosol_Cytoplasmic.tsv')
    centriolar_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Centriolar.tsv')
    endoplasmic_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Endoplasmic.tsv')
    golgi_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Golgi.tsv')
    intermediate_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Intermediate.tsv')
    microtubule_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Microtubules_Microtubule.tsv')
    mitochondria_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Mitochondria.tsv')
    plasma_prots = get_subcell_prots('./Data/subcellular_locations/subcell_location_Plasma.tsv')
    vesicle_prots = get_subcell_prots(
        './Data/subcellular_locations/subcell_location_Vesicles_Peroxisomes_Endosomes_Lysosomes_Lipid.tsv')

    tot_prots = np_prots.union(nuclear_membrane_prots,
                               nucleoli_prots,
                               actin_prots,
                               aggresome_prots,
                               centriolar_prots,
                               endoplasmic_prots,
                               golgi_prots,
                               intermediate_prots,
                               microtubule_prots,
                               mitochondria_prots,
                               plasma_prots,
                               vesicle_prots,
                               )

    subcell_prot_dict = {}
    subcell_loc_dict = {'nuclear_plasma': set(),
                        'nuclear_membrane': set(),
                        'nucleoli': set(),
                        'actin': set(),
                        'aggresome': set(),
                        'centriolar': set(),
                        'endoplasmic': set(),
                        'golgi': set(),
                        'intermediate': set(),
                        'microtuble': set(),
                        'mitochondria': set(),
                        'plasma': set(),
                        'vesicle': set(),
                        }

    for prot in tot_prots:
        sub_cell_list = set()
        if prot in np_prots:
            sub_cell_list.add('nuclear_plasma')
            subcell_loc_dict['nuclear_plasma'].add(prot)
        if prot in nuclear_membrane_prots:
            sub_cell_list.add('nuclear_membrane')
            subcell_loc_dict['nuclear_membrane'].add(prot)
        if prot in nucleoli_prots:
            sub_cell_list.add('nucleoli')
            subcell_loc_dict['nucleoli'].add(prot)
        if prot in actin_prots:
            sub_cell_list.add('actin')
            subcell_loc_dict['actin'].add(prot)
        if prot in aggresome_prots:
            sub_cell_list.add('aggresome')
            subcell_loc_dict['aggresome'].add(prot)
        if prot in centriolar_prots:
            sub_cell_list.add('centriolar')
            subcell_loc_dict['centriolar'].add(prot)
        if prot in endoplasmic_prots:
            sub_cell_list.add('endoplasmic')
            subcell_loc_dict['endoplasmic'].add(prot)
        if prot in golgi_prots:
            sub_cell_list.add('golgi')
            subcell_loc_dict['golgi'].add(prot)
        if prot in intermediate_prots:
            sub_cell_list.add('intermediate')
            subcell_loc_dict['intermediate'].add(prot)
        if prot in microtubule_prots:
            sub_cell_list.add('microtuble')
            subcell_loc_dict['microtuble'].add(prot)
        if prot in mitochondria_prots:
            sub_cell_list.add('mitochondria')
            subcell_loc_dict['mitochondria'].add(prot)
        if prot in plasma_prots:
            sub_cell_list.add('plasma')
            subcell_loc_dict['plasma'].add(prot)
        if prot in vesicle_prots:
            sub_cell_list.add('vesicle')
            subcell_loc_dict['vesicle'].add(prot)

        subcell_prot_dict[prot] = sub_cell_list

    return subcell_prot_dict, subcell_loc_dict


# convert gene dictionary into readable gene names
def convert_gene_dict(gene_dict, name_convention='an'):
    converted_gene_dict = {}
    for key, sub_dict in gene_dict.items():
        if name_convention == 'a':
            new_key = str(sub_dict['Accession'])
        elif name_convention == 'g':
            new_key = str(sub_dict['Name'])
        elif name_convention == 'name':
            new_key = str(key)
            key = str(sub_dict['Name'])
        elif name_convention == 'accession':
            new_key = str(key)
            key = str(sub_dict['Accession'])
        elif name_convention == 'na':
            new_key = sub_dict['Name']
            key = str(sub_dict['Accession'])
        elif name_convention == 'an':
            new_key = sub_dict['Accession']
            key = str(sub_dict['Name'])

        converted_gene_dict[str(new_key)] = str(key)

    return converted_gene_dict


# creates a network based on a dictionary with a list of 5 values
# takes the dictionary and which time trials as parameters
def draw_network(dic, numList, func=np.mean):
    G = nx.Graph()
    for key,value in dic.items():
        tempList = []
        for element in numList:
            tempList.append(value[element])
        G.add_edge(key[0], key[1], weight=func(tempList))
    nx.draw(G, node_size=20, width=0.1)
    return G


# creates communities of 10+ nodes of an original network
def draw_community(network, Int):
    comp = nx.community.girvan_newman(network)
    communities = next(comp)
    x = network.subgraph([comm for comm in communities if len(comm)>25][Int])
    nx.draw(x, node_size=20, width=0.1)
    return x


# finds the betweenness centrality of every unique protein
def nx_centrality(dic, num):
    new_dic = {}
    for key, value in dic.items():
        new_dic[key] = value[num]

    G = nx.Graph()
    for key, value in new_dic.items():
        G.add_edge(key[0], key[1], weight=value)

    btw = nx.betweenness_centrality(G, weight="weight")
    sorted_keys = btw.items()
    new_values = sorted(sorted_keys)
    finalDict = {}
    for element in new_values:
        finalDict[element[0]] = element[1]
    return finalDict.values()


# formats a list of nodes
def format_nodes(nodeList):
    for element in nodeList:
        print(element)


# takes a dictionary, returns a list of numbers
# finds the amount of shared connections with other proteins each protein pair has
def find_shared_connections(dic):
    lenList = []
    for key in dic.keys():
        key0List = []
        key1List = []
        for keys in dic.keys():
            if key[0] in keys:
                if keys[0] == key[0]:
                    key0List.append(keys[1])
                else:
                    key0List.append(keys[0])
            if key[1] in keys:
                if keys[1] == key[1]:
                    key1List.append(keys[0])
                else:
                    key1List.append(keys[1])
        count = 0
        for element in key0List:
            if element in key1List:
                count = count + 1
        lenList.append(count)

    return lenList


# finds the number of connections each protein has with all the other proteins
def number_connections(dic):
    numList = []
    uni_prots = list(f.find_unique_prots(dic))
    uni_prots.sort()

    for element in uni_prots:
        count = 0
        for key in dic.keys():
            if element in key:
                count = count + 1
        numList.append(count)

    return numList


# finds the average of all the connections for all the unique proteins at various time points
def average_connections(dic, hourList):
    avgList = []
    uni_prots = list(f.find_unique_prots(dic))
    uni_prots.sort()

    for element in uni_prots:
        valList = []
        for key, value in dic.items():
            if element in key:
                for i in hourList:
                    valList.append(value[i])
        avgList.append(np.mean(valList))

    return avgList


# finds the median of all the connections for all the unique proteins at various time points
def median_connections(dic, hourList):
    medList = []
    uni_prots = list(f.find_unique_prots(dic))
    uni_prots.sort()

    for element in uni_prots:
        valList = []
        for key, value in dic.items():
            if element in key:
                for i in hourList:
                    valList.append(value[i])
        medList.append(np.median(valList))

    return medList


# finds the range of all the connections for all the unique proteins at various time points
def range_connections(dic, hourList):
    rangeList = []
    uni_prots = list(f.find_unique_prots(dic))
    uni_prots.sort()

    for element in uni_prots:
        valList = []
        for key, value in dic.items():
            if element in key:
                for i in hourList:
                    valList.append(value[i])
        rangeList.append(np.max(valList) - np.min(valList))

    return rangeList


# creates a dataframe using the dictionary of f.open_file2(location)
def create_df(dic):
    df = pd.DataFrame(dic).T
    df.columns = ['Hour_0', 'Hour_1', 'Hour_3', 'Hour_8', 'Hour_15']
    df.insert(0, "Protein Pairs", dic.keys(), True)
    df.insert(6, "Shared Connections", f.find_shared_connections(dic),True)
    return df


# creates a violin plot using the dictionary of f.open_file2(location)
def create_violinplot(dic):
    df = create_df(dic)
    fig, ax = plt.subplots(figsize=[10,10])
    ax.violinplot([df['Hour_0'], df['Hour_1'], df['Hour_3'], df['Hour_8'], df['Hour_15']], showmedians=True)
    ax.set_title('Strength of Connections (%) VS Time (Hrs)')
    ax.set_ylabel('Chance of Interacting (%)')
    ax.set_xlabel('Time (Hrs)')
    plt.show()
    return df


# creates a scatter plot using the dictionary of f.open_file2(location)
def create_scatterplot(dic):
    df = create_df(dic)
    fig = make_subplots(rows=1, cols=5, subplot_titles=("Hour 0", "Hour 1", "Hour 3", "Hour 8", "Hour 15"))

    fig.add_trace(go.Scatter(x=df.loc[:, "Shared Connections"], y=df.loc[:, "Hour_0"], mode="markers",
                             hovertext=list(df.loc[:, "Protein Pairs"]),
                             marker=dict(color=df.loc[:, "Hour_0"], coloraxis="coloraxis")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.loc[:, "Shared Connections"], y=df.loc[:, "Hour_1"], mode="markers",
                             hovertext=list(df.loc[:, "Protein Pairs"]),
                             marker=dict(color=df.loc[:, "Hour_0"], coloraxis="coloraxis")), row=1, col=2)
    fig.add_trace(go.Scatter(x=df.loc[:, "Shared Connections"], y=df.loc[:, "Hour_3"], mode="markers",
                             hovertext=list(df.loc[:, "Protein Pairs"]),
                             marker=dict(color=df.loc[:, "Hour_0"], coloraxis="coloraxis")), row=1, col=3)
    fig.add_trace(go.Scatter(x=df.loc[:, "Shared Connections"], y=df.loc[:, "Hour_8"], mode="markers",
                             hovertext=list(df.loc[:, "Protein Pairs"]),
                             marker=dict(color=df.loc[:, "Hour_0"], coloraxis="coloraxis")), row=1, col=4)
    fig.add_trace(go.Scatter(x=df.loc[:, "Shared Connections"], y=df.loc[:, "Hour_15"], mode="markers",
                             hovertext=list(df.loc[:, "Protein Pairs"]),
                             marker=dict(color=df.loc[:, "Hour_0"], coloraxis="coloraxis")), row=1, col=5)

    fig.update_xaxes(title_text="Shared Connections", row=1, col=1)
    fig.update_xaxes(title_text="Shared Connections", row=1, col=2)
    fig.update_xaxes(title_text="Shared Connections", row=1, col=3)
    fig.update_xaxes(title_text="Shared Connections", row=1, col=4)
    fig.update_xaxes(title_text="Shared Connections", row=1, col=5)

    fig.update_yaxes(title_text="Interaction Chance (%)", row=1, col=1)
    fig.update_yaxes(title_text="Interaction Chance (%)", row=1, col=2)
    fig.update_yaxes(title_text="Interaction Chance (%)", row=1, col=3)
    fig.update_yaxes(title_text="Interaction Chance (%)", row=1, col=4)
    fig.update_yaxes(title_text="Interaction Chance (%)", row=1, col=5)

    fig.update_annotations(font_size=24)
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(height=1000, width=6000, font_size=24,
                      title_text="Interaction Chance (%) VS Shared Connections Graphs", showlegend=False,
                      coloraxis=dict(colorscale='bluered'), font_family="Times New Roman", font_color="black",
                      title_font_family="Times New Roman", title_font_color="black")
    fig.show()
    # fig.write_html("./Data/ScatterGraph.html")
    return df