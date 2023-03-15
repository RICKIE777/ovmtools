#!/usr/bin/python3

import sys, os
import argparse
import os.path
import json
import compare_vis

exec_graph_A = ''
exec_graph_B = ''
pc_log_start_tag = "[ INFO ] Performance counts for 0-th infer request:"
pc_log_end_tag = "Total time:"


def show_compare_result(log_file_A, log_file_B, all_dict, prefixA, prefixB, reportA, reportB):
    my_verbose_converter = compare_vis.set_verbose()
    pc_by_node0, pc_by_type0, stat0, verbose_by_name0, statis_by_type0 = compare_vis.analyse(log_file_A, reportA)
    if prefixB:
        pc_by_node1, pc_by_type1, stat1, verbose_by_name1, statis_by_type1 = compare_vis.analyse(log_file_B, reportB)
    else:
        pc_by_node1 = []
        pc_by_type1 = []
        stat1 = []
        verbose_by_name1 = ""
        statis_by_type1 = []
    print("{}   :    {}".format(log_file_A, log_file_B))

    print("*********************************************************")
    print("*                   comparing by node                   *")
    print("*********************************************************")
    # collect all type names
    all_names = list(set([t for t, _ in pc_by_node0])|set([t for t, _ in pc_by_node1]))

    total_time0 = total_time1 = 0
    layerid = 0
    for name in all_names:
        layer_dict={}
        layer_info = []
        A_info = []
        B_info = []
        time_diff = 0
        info0, time0, layer0, exectype0, layout0 = get_exec_info(pc_by_node0, name)  # prefixA
        info1, time1, layer1, exectype1, layout1 = get_exec_info(pc_by_node1, name)  # prefixB
        layer_info = [layerid, name]
        A_info = [layer0, exectype0, layout0, time0]
        B_info = [layer1, exectype1, layout1, time1]
        node_type = ""
        show_verbose= True
        if node_type in info0 or node_type in info1:
            total_time0 += time0
            total_time1 += time1
            time_diff = compare_vis.smart_val(time1 - time0)
            print("{:>6} {:>50}  {:<50}  {}".format(time_diff, info0, info1, name))

            if show_verbose:
                verbose0, benchdnn0 = get_print_info(my_verbose_converter, name, verbose_by_name0)
                verbose1, benchdnn1 = get_print_info(my_verbose_converter, name, verbose_by_name1)
                A_info += [verbose0, benchdnn0]
                B_info += [verbose1, benchdnn1]
        layerid += 1
        dict_data = JsonData(prefixA, prefixB)
        layer_dict = dict_data.layer(layer_info, A_info, B_info, time_diff)
        all_dict.update(layer_dict)


    print("")
    print("{:>6} {:>50}   {:<50}   {}".format(compare_vis.smart_val(total_time1 - total_time0), total_time0,total_time1, "Totals"))

    print("")
    for i in range(len(stat0)):
        s0 = stat0[i].rstrip("\n").rstrip("\r")
        if stat1: s1 = stat1[i].rstrip("\n").rstrip("\r")
        else: s1 = "None"
        print("{:>50}   {:<50} ".format(s0, s1))

    return all_dict


def get_exec_info(pc_by_node, name):
    def find(pclist, type_name):
        for name, v in pclist:
            if name == type_name:
                return v
        return None
    v0 = find(pc_by_node, name)
    if v0:
        time0, layer0, exectype = v0
        layout = compare_vis.find_layout(exec_graph_A, name)
        info0 = "{}_{}_{} {:6.1f}".format(layer0, exectype, layout, time0)
    else:
        time0 = 0
        info0 = "---"
        layer0 = ""
        exectype = ""
        layout = ""
    return info0, time0, layer0, exectype, layout


def get_print_info(my_verbose_converter, name, verbose_by_name0):
    verbose = ""
    benchdnn = ""
    if name in verbose_by_name0:
        verbose = 'onednn_verbose,exec,' + verbose_by_name0[name]
        print(verbose)
        all_verbose = \
f'''
onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
{verbose},1.7478
'''
        if my_verbose_converter:
            status, output = my_verbose_converter(verbose_level=0, parser='oneDNN',
                                                  input=all_verbose.splitlines(), action='generate',
                                                  generator='benchdnn', split_output=False, agg_keys=None)
            if output != None:
                for key, value in output.items():
                    benchdnn = f"./benchdnn --fix-times-per-prb=100 --mode=p {value}"
                    print(f"./benchdnn --fix-times-per-prb=100 --mode=p {value}", end='')
    return verbose, benchdnn


def write_json_data(all_dict, output_path, model):
    name = get_output_name(output_path, model)
    json_str = json.dumps(all_dict, indent=2)
    with open(os.path.join(output_path, name + ".json"), "w+") as f:
        f.write(json_str)


def get_output_name(output_path, model):
    path = model.split("/")
    name = path[-1].split(".")[0]
    while (os.path.exists(os.path.join(output_path, name + ".json"))):
        name = name + "_1"
    return name


# layer_info = [layer_id, layer_name]
# A_info = [node_type, exec_type, layout, time, onednn_cmd, benchdnn_cmd]
# B_info = [node_type, exec_type, layout, time, onednn_cmd, benchdnn_cmd]
# time_diff the time difference between brgconv and jit


class JsonData:
    def __init__(self, prefixA, prefixB):
        self.jsonData = {}
        self.prefixA = prefixA
        self.prefixB = prefixB

    def layer(self, layer_info, A_info, B_info, time_diff):
        layer_id = "layer" + str(layer_info[0])
        self.jsonData[layer_id] = {}
        self.jsonData[layer_id]["layer_name"] = layer_info[1]
        self.jsonData[layer_id][self.prefixA] = self.exec_node(A_info)
        self.jsonData[layer_id][self.prefixB] = self.exec_node(B_info)
        self.jsonData[layer_id]["time_difference"] = time_diff
        return self.jsonData

    @staticmethod
    def exec_node(info):
        exec_dict = {}
        exec_dict["node_type"] = info[0]
        exec_dict["exec_type"] = info[1]
        exec_dict["layout"] = info[2]
        exec_dict["time"] = info[3]
        exec_dict["onednn_cmd"] = info[4]
        exec_dict["benchdnn_cmd"] = info[5]
        return exec_dict


def main(exec_graph_A, exec_graph_B, model, log_file_A, log_file_B, prefixA, prefixB, reportA, reportB, output_file):
    try:
        with open(exec_graph_A or './a/exec_graph_A.xml') as f:
            print("prefixA: ", exec_graph_A)
            exec_graph_A = f.readlines()
    except FileNotFoundError:
        exec_graph_A = ""

    try:
        with open(exec_graph_B or './b/exec_graph_B.xml') as f:
            print("prefixB: ", exec_graph_B)
            exec_graph_B = f.readlines()
    except FileNotFoundError:
        exec_graph_B = ""

    all_dict = {"model_name": model}
    data_cmp = show_compare_result(log_file_A or './testA.log', log_file_B or './testB.log',
                                   all_dict, prefixA, prefixB, reportA, reportB)
    write_json_data(data_cmp, output_file or './output', model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--node_cnt", help="number of nodes to show", default=1000, type=int)
    parser.add_argument("-t", "--node_type", help="node type filter", default="", type=str)
    parser.add_argument("log_file_A", nargs="?")
    parser.add_argument("log_file_B", nargs="?")
    parser.add_argument("exec_graph_A", nargs="?")
    parser.add_argument("exec_graph_B", nargs="?")
    parser.add_argument("-s", "--show_verbose", default=True, help="show onednn verbose",
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("-m", "--model", default="", type=str)
    parser.add_argument("-output_file", default="", type=str)
    parser.add_argument("-rA", "--reportA", help="report folderA", default="./a", type=str)
    parser.add_argument("-rB", "--reportB", help="report folderB", default="./b", type=str)
    parser.add_argument("-pA", "--prefixA", help="prefixA", default="brg", type=str)
    parser.add_argument("-pB", "--prefixB", help="prefixB", default="jit", type=str)
    args = parser.parse_args()

    main(args.exec_graph_A, args.exec_graph_B, args.model, args.log_file_A, args.log_file_B,
         args.prefixA, args.prefixB, args.reportA, args.reportB, args.output_file)

    print("finish")
