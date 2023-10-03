import pandas as pd
import numpy as np
import os
import pathlib
from graphtempo import *
import itertools
import copy
import time
from collections import defaultdict
from collections import OrderedDict
import collections
import ast


# distinct AGGREGATION for PROPERTY GRAPHS | STATIC

def Intersection_Static(nodesdf,edgesdf,tia,intvl):
    # get intersection of nodes and edges on interval
    nodes_inx = nodesdf[intvl][nodesdf[intvl].all(axis=1)]
    edges_inx = edgesdf[intvl][edgesdf[intvl].all(axis=1)]
    tia_inx = tia[tia.index.isin(nodes_inx.index)]
    inx = [nodes_inx,edges_inx]
    return(inx,tia_inx)


def Aggregate_Static_Dist_PG(inx,tia_inx,stc_attrs):
    types_to_choose = list(inx[0].index.get_level_values('type').value_counts().index)
    type_to_agg = ['author', 'conference']
    
    if inx[0].index.equals(tia_inx.index):
        nodes = tia_inx[stc_attrs].set_index(tia_inx[stc_attrs].columns.values.tolist())
    else:#difference output produces different indexes for nodes and attributes
        nodes = pd.DataFrame(index=inx[0].index)
        for attr in stc_attrs:
            nodes[attr] = tia_inx.loc[nodes.index,attr].values
    
    nodes_orig_not_agg = set([i[0] for i in nodes.index if i[1] not in type_to_agg])
    nodes = nodes.set_index(stc_attrs, append=True)
    idx_agg = [i for i in nodes.index if i[1] in type_to_agg]
    idx_not_agg = [i for i in nodes.index if i[1] not in type_to_agg]
    nodes_agg = nodes.loc[idx_agg]
    nodes_agg = nodes_agg.droplevel('id')
    nodes_agg = nodes_agg.groupby(nodes_agg.index.names).size().to_frame('count')
    nodes_not_agg = nodes.loc[idx_not_agg]
    nodes_not_agg = nodes_not_agg.droplevel(stc_attrs)
    nodes_not_agg = nodes_not_agg.swaplevel()
    nodes_not_agg['count'] = 1
    nodes_all = pd.concat([nodes_agg, nodes_not_agg], axis=0)
    
    # edges
    edges = pd.DataFrame(index=inx[1].index)
    edges_idx = edges.index
    
    for attr in stc_attrs:
        edges[attr+'L'] = \
        tia_inx.loc[edges.index.get_level_values('Left'),attr].values
    for attr in stc_attrs:
        edges[attr+'R'] = \
        tia_inx.loc[edges.index.get_level_values('Right'),attr].values
    for attr in stc_attrs:
        for idx in edges_idx:
            if idx[0] in nodes_orig_not_agg:
                edges.loc[idx, attr+'L'] = idx[0]
            if idx[1] in nodes_orig_not_agg:
                edges.loc[idx, attr+'R'] = idx[1]
    edges = edges.droplevel(['Left','Right'])
    edges = edges.set_index(edges.columns.values.tolist(), append=True)
    edges = edges.groupby(edges.index.names).size().to_frame('count')
    agg = [nodes_all, edges]
    return(agg)

# read csv
nodes_df = pd.read_csv('datasets/DBLP/dblp_prop_graph/nodes_df.csv', sep=' ', index_col=[0,1])
edges_df = pd.read_csv('datasets/DBLP/dblp_prop_graph/edges_df.csv', sep=' ', index_col=[0,1,2])
time_invar = pd.read_csv('datasets/DBLP/dblp_prop_graph/time_invar.csv', sep=' ', index_col=[0,1], dtype=str)
time_var = pd.read_csv('datasets/DBLP/dblp_prop_graph/time_var.csv', sep=' ', index_col=[0,1], dtype=str)


#############################################################

attr_val_left = time_invar.gender[time_invar.index.get_level_values('type') == 'author'].unique().tolist()
#attr_val_right = time_invar[time_invar.index.get_level_values('type') == 'conference'].index.get_level_values('id').tolist()
attr_val_right = time_invar.topic[time_invar.index.get_level_values('type') == 'conference'].unique().tolist()

attr_val_combs_coll = list(itertools.product(attr_val_left, attr_val_left))
attr_val_combs_publ = list(itertools.product(attr_val_left, attr_val_right))


c=0
intvls = []
for i in range(1,len(edges_df.columns)+1-c):
    intvls.append([list(edges_df.columns[:i]), list(edges_df.columns[i:i+1])])
    c += 1
intvls = intvls[:-1]

edge_type = 'publication'
stc_attrs = ['gender', 'topic']

edge_type = 'collaboration'
stc_attrs = ['gender']


# 1. UNIFIED / ONE-PASS SKYLINE EXPLORATION

# SKYLINE ONE-PASS | STATIC

skyline_st = []
dominate_counter = {}
for left,right in intvls:
    #print('left, right: ', left, right)
    while len(left) >= 1:
        current_w = []
        inx,tia_inx = Intersection_Static(nodes_df,edges_df,time_invar,left+right)
        if not inx[1].empty:
            #agg_inx = Aggregate_Mix_Dist_PG(inx,tva_inx,tia_inx,stc_attrs,left+right)
            agg_inx = Aggregate_Static_Dist_PG(inx,tia_inx,stc_attrs)
            edges_sky = agg_inx[1][agg_inx[1].index.get_level_values('type') == edge_type].droplevel('type')
            edges_sky_idx = [tuple([i for i in tpl if i!='0']) for tpl in edges_sky.index]
            for comb in attr_val_combs_publ:
            #for comb in attr_val_combs_coll:
                if comb in edges_sky_idx:
                    result = edges_sky.loc[(comb[0],'0', '0', comb[1]),:][0]#publ
                    #result = edges_sky.loc[(comb),:][0]#coll
                else:
                    result = 0
                current_w.append(result)
            if all([w==0 for w in current_w]):
                left = left[1:]
                continue
            current_w.append(len(left))
            dominate_counter[str((current_w,left,right))] = 0
            if not skyline_st:
                skyline_st.append([current_w,left,right])
            else:
                print(left)
                flags = []
                sky_rm = []
                for sky in skyline_st:
                    if all([current_w[i] >= sky[0][i] for i in range(len(current_w))]) and \
                        any([current_w[i] > sky[0][i] for i in range(len(current_w))]):
                        dominate_counter[str((current_w,left,right))] += 1
                        dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(sky))]
                        dominate_counter[str(tuple(sky))] = 0
                        #skyline_st[:] = [s for s in skyline_st if s != sky]
                        sky_rm.append(sky)
                        flags.append(1)
                    # tie
                    elif any([current_w[i] > sky[0][i] for i in range(len(current_w))]) or \
                        all([current_w[i] == sky[0][i] for i in range(len(current_w))]):
                        flags.append(2)
                    # curr dominated by sky
                    elif all([current_w[i] <= sky[0][i] for i in range(len(current_w))]) and \
                        any([current_w[i] < sky[0][i] for i in range(len(current_w))]):
                        dominate_counter[str(tuple(sky))] += 1
                        flags.append(3)
                skyline_st = [sky for sky in skyline_st if sky not in sky_rm]
                if 3 not in flags:
                    skyline_st.append([current_w,left,right])
        left = left[1:]


###############################################################################
# 2. K-DOMINANT SKYLINE EXPLORATION

# CREATE DATASET
# SORT TUPLES ACCORDING TO LENGTH
# SORT TUPLES ACCORDING TO SUM OF TUPLE'S DIMENSIONS

dataset = defaultdict(dict)
tuples_dict = {}
for left,right in intvls:
    #print('left, right: ', left, right)
    while len(left) >= 1:
        inx,tia_inx = Intersection_Static(nodes_df,edges_df,time_invar,left+right)
        if not inx[1].empty:
            #agg_inx = Aggregate_Mix_Dist_PG(inx,tva_inx,tia_inx,stc_attrs,left+right)
            agg_inx = Aggregate_Static_Dist_PG(inx,tia_inx,stc_attrs)
# =============================================================================
#             tpl = [agg_inx[1].droplevel('type')
#                    .loc[i,:][0] if i in agg_inx[1]
#                    .droplevel('type').index else 0 for i in attr_val_combs_coll]
# =============================================================================
            agg_edges = agg_inx[1].droplevel(['type', 'topicL', 'genderR'])
            agg=agg_edges[agg_edges.index.isin(attr_val_right, level=1)]
            agg = agg['count'].to_dict()
            tpl = [agg[i] if i in agg.keys() else 0 for i in attr_val_combs_publ]
            if all([dim==0 for dim in tpl]):
                left = left[1:]
                continue
            # add length to tpl
            length = len(left)
            tpl.append(length)
            tuples_dict.setdefault(length,[]).append(tpl)
            dataset[length][str([left,right])] = tpl
        left = left[1:]


# NORMALIZATION

norm_array = []
for k,d in dataset.items():
    tmp = []
    for key,val in d.items():
        tmp.append(val)
    norm_array.extend(tmp)
norm_array = np.array(norm_array)

max_values = norm_array.max(axis=0)
min_values = norm_array.min(axis=0)

dataset_norm = defaultdict(dict)
for key,d in dataset.items():
    for k,v in d.items():
        dimensions = []
        for i in range(len(v)):
            if (max_values[i] - min_values[i]) == 0:
                dimension_norm = 0
            else:
                dimension_norm = (v[i] - min_values[i])/(max_values[i] - min_values[i])
            dimension_norm = format(dimension_norm, '.2f')
            dimensions.append(dimension_norm)
        dataset_norm[key][k] = dimensions


# SORTING --> LENGTH & SUM OF VALUES

# sort without normalization
dataset_to_sort = copy.deepcopy(dataset)
# sort normalized dataset
#dataset_to_sort = copy.deepcopy(dataset_norm)
dataset_keys = sorted(dataset_to_sort.keys())[::-1]
dataset_descending = [dataset_to_sort[i] for i in dataset_keys]
dataset_sorted = []
for d in dataset_descending:
    for key,value in sorted(d.items(),key=lambda i:sum(i[1]),reverse=True):
        tmp = ast.literal_eval(key)
        dataset_sorted.append([value,tmp[0],tmp[1]])

dataset_no_sort = [[val,ast.literal_eval(key)[0],ast.literal_eval(key)[1]] \
                   for k,d in dataset.items() for key,val in d.items()]

###############################################################################

# 2.1 K-DOMINANT SKYLINE EXPLORATION

k=33
# 1-SCAN
skyline_st = []
dominated_tpls = []
for (tpl,left,right) in dataset_sorted:
    if not skyline_st:
        skyline_st.append([tpl, left, right])
    else:
        is_dominated = False
        skyline_rm = []
        for ind,tpl_sky in enumerate(skyline_st):
            k_cand = tpl_sky[0]
            k_tpl_greateq = 0
            k_tpl_great = 0
            for idx in range(len(tpl)):
                if tpl[idx] >= k_cand[idx]:
                    k_tpl_greateq += 1
                if tpl[idx] > k_cand[idx]:
                    k_tpl_great += 1
            if k_tpl_greateq >= k and k_tpl_great >= 1:
                skyline_rm.append(skyline_st[ind])
            if is_dominated == False:
                k_cand_greateq = 0
                k_cand_great = 0
                for idx in range(len(tpl)):
                    if k_cand[idx] >= tpl[idx]:
                        k_cand_greateq += 1
                    if k_cand[idx] > tpl[idx]:
                        k_cand_great += 1
                if k_cand_greateq >= k and k_cand_great >= 1:
                    is_dominated = True
                    if [tpl, left, right] not in dominated_tpls:
                        dominated_tpls.append([tpl, left, right])
        skyline_st = [i for i in skyline_st if i not in skyline_rm]
        if is_dominated == False:
            skyline_st.append([tpl, left, right])

# 2-SCAN
skyline_rm = []
for tpl_dom in dominated_tpls:
    tpl = tpl_dom[0]
    for ind,tpl_sky in enumerate(skyline_st):
        k_cand = tpl_sky[0]
        k_tpl_greateq = 0
        k_tpl_great = 0
        for idx in range(len(tpl)):
            if tpl[idx] >= k_cand[idx]:
                k_tpl_greateq += 1
            if tpl[idx] > k_cand[idx]:
                k_tpl_great += 1
        if k_tpl_greateq >= k and k_tpl_great >= 1:
            skyline_rm.append(skyline_st[ind])
skyline_st = [i for i in skyline_st if i not in skyline_rm]


###############################################################################

# 2.2 LENGTH RESTRICTED K-DOMINANT SKYLINE EXPLORATION

k=33
# 1-SCAN
skyline_st = []
dominated_tpls = []
for (tpl,left,right) in dataset_sorted:
    if not skyline_st:
        skyline_st.append([tpl, left, right])
    else:
        is_dominated = False
        skyline_rm = []
        for ind,tpl_sky in enumerate(skyline_st):
            k_cand = tpl_sky[0]
            if tpl[-1] >= k_cand[-1]:
                k_tpl_greateq = 0
                k_tpl_great = 0
                for idx in range(len(tpl)):
                    if tpl[idx] >= k_cand[idx]:
                        k_tpl_greateq += 1
                    if tpl[idx] > k_cand[idx]:
                        k_tpl_great += 1
                if k_tpl_greateq >= k and k_tpl_great >= 1:
                    skyline_rm.append(skyline_st[ind])
            if is_dominated == False:
                if k_cand[-1] >= tpl[-1]:
                    k_cand_greateq = 0
                    k_cand_great = 0
                    for idx in range(len(tpl)):
                        if k_cand[idx] >= tpl[idx]:
                            k_cand_greateq += 1
                        if k_cand[idx] > tpl[idx]:
                            k_cand_great += 1
                    if k_cand_greateq >= k and k_cand_great >= 1:
                        is_dominated = True
                        if [tpl, left, right] not in dominated_tpls:
                            dominated_tpls.append([tpl, left, right])
        skyline_st = [i for i in skyline_st if i not in skyline_rm]
        if is_dominated == False:
            skyline_st.append([tpl, left, right])

# 2-SCAN
skyline_rm = []
for tpl_dom in dominated_tpls:
    tpl = tpl_dom[0]
    for ind,tpl_sky in enumerate(skyline_st):
        k_cand = tpl_sky[0]
        if tpl[-1] >= k_cand[-1]:
            k_tpl_greateq = 0
            k_tpl_great = 0
            for idx in range(len(tpl)):
                if tpl[idx] >= k_cand[idx]:
                    k_tpl_greateq += 1
                if tpl[idx] > k_cand[idx]:
                    k_tpl_great += 1
            if k_tpl_greateq >= k and k_tpl_great >= 1:
                skyline_rm.append(skyline_st[ind])
skyline_st = [i for i in skyline_st if i not in skyline_rm]
