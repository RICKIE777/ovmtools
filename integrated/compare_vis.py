#!/usr/bin/python3

import sys, os
import argparse
import os.path
import json


def set_verbose():
    my_verbose_converter = None
    if 'VERBOSE_CONVERT' in os.environ:
        sys.path.append(os.environ['VERBOSE_CONVERT'])
        try:
            import verbose_converter

            my_verbose_converter = verbose_converter.convert
            return my_verbose_converter
        except Exception as e:
            print(e)
            pass

exec_graphA = ''
exec_graphB = ''
args = None


def find_layout(exec_graph, name):
    if (name.endswith("...")):
        tag = 'originalLayersNames="{}'.format(name.rstrip("..."))
    else:
        tag = 'originalLayersNames="{}'.format(name)
    layout='outputLayouts="'
    found = 0
    ret = "?"
    for l in exec_graph:
        if tag in l:
            sl = l[l.index(layout) + len(layout):]
            found += 1
            ret = sl[0:sl.index('"')]
    
    if found > 1:
        ret = "?"
    return ret

if args and len(args.node_type) > 0 and args.node_cnt <= 0:
    args.node_cnt = 99999999

pc_log_start_tag = "[ INFO ] Performance counts for 0-th infer request:"
pc_log_end_tag="Total time:"


class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"

def analyse(log_file, json_dir):
    pc_by_type = {}
    pc_by_node = {}
    statis_by_type = {}
    
    stat = []
    verbose_by_name = {}
    layers = None
    if os.path.isfile(f'{json_dir}/benchmark_detailed_counters_report.json'):
        with open(f'{json_dir}/benchmark_detailed_counters_report.json',"r") as f:
            layers = json.loads(f.read())

    def append_to_result(run, layer_type,realTime, cpuTime, execType, name):
        if (run == "NOT_RUN"):
            return
        node_type = layer_type + "_" + execType

        if node_type.startswith('Convolution_brgconv_avx512'):
            node_type = node_type.replace('brgconv', 'brg.jit')
        elif node_type.startswith('Convolution_jit_avx512'):
            node_type = node_type.replace('jit', 'brg.jit')
        elif node_type.startswith('Convolution_jit_gemm_FP32'):
            node_type = 'Convolution_brg.jit_avx512_FP32'
        if node_type.startswith('GroupConvolution_ref_any_FP32'):
            node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'
        elif node_type.startswith('GroupConvolution_brgconv_avx512') and node_type.endswith('_FP32'):
            node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'
        elif node_type.startswith('GroupConvolution_jit_avx512_FP32'):
            node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'
        elif node_type.startswith('GroupConvolution_jit_gemm_FP32'):
            node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'

        if not node_type in pc_by_type:
            pc_by_type[node_type] = [0,0] # cnt, total
        if layer_type not in statis_by_type:
            statis_by_type[layer_type] = [0, 0]
        pc_by_type[node_type][0] += 1
        pc_by_type[node_type][1] += round(float(realTime))
        statis_by_type[layer_type][0] += 1
        statis_by_type[layer_type][1] += round(float(realTime))

        pc_by_node[name] = [round(float(realTime)), layer_type, execType]
    with open(log_file,"r") as f:
        start  = False
        for l in f.readlines():
            if 'verbose##' in l:
                items = l.split('##')
                verbose_by_name[items[1]] = items[2].strip()
                continue
            if l.startswith("	Percent of CPU this job got") or \
               l.startswith("	Maximum resident set size (kbytes)") or \
               l.startswith("	User time (seconds)"):
                stat.append(l.strip("\t"))
                continue
            if l.startswith("[ INFO ] 	Average:") or \
               l.startswith("[ INFO ] Throughput:") or \
               l.startswith("[ INFO ]    Average:"):
                stat.append(l[9:].strip(" ").strip("\t"))
                continue

            if l == '\n' or layers: continue
            if l.startswith(pc_log_start_tag):
                start = True
                continue
            if start:
                if l.startswith(pc_log_end_tag):
                    start = False
            if start:
                if "(ms)" in l:
                    name = l[:21].rstrip(" ")
                    run, _, layer_type, _, execType, _, _, realTime, _, _, cpuTime = l[21:].split()
                else:
                    name = l[:30].rstrip(" ")
                    run, _, layer_type, _, realTime, _, cpuTime, _, execType = l[30:].split()
                append_to_result(run, layer_type, realTime, cpuTime, execType, name)

    if layers:
        for layer in layers['detailed_performance'][0]['nodes']:
            append_to_result(layer['status'], layer['node_type'],\
                layer['real_time']*1000, layer['cpu_time']*1000, layer['exec_type'], layer['name'])

    pc_by_node = sorted(pc_by_node.items(), key=lambda d: d[1][0], reverse=True)
    pc_by_type = sorted(pc_by_type.items(), key=lambda d: d[1][1], reverse=True)
    statis_by_type = sorted(statis_by_type.items(), key=lambda d: d[1][1], reverse=True)
    return pc_by_node, pc_by_type, stat, verbose_by_name, statis_by_type


def show_result(log_file, pc_by_node, pc_by_type, stat):
    if args.node_cnt > 0:
        print("=========== {} by node =============".format(log_file)) 
        for i, (name, (total, *other)) in enumerate(pc_by_node):
            if i >= args.node_cnt:
                break
            if total == 0:
                break
            print("{:>8} us {} {}".format(total, name, other))

    total_time = 0
    print("=========== {} by type =============".format(log_file)) 
    for type_name, (cnt, total) in pc_by_type:
        total_time += total
        if total == 0:
            break
        print("{:>8.1f} x {:<5} us {}".format(total/cnt, cnt, type_name))

    print("total_time:{}".format(total_time))
    for s in stat:
        print(s.rstrip("\n").rstrip("\r"))


def smart_val(v):
    if abs(v) > 1000000:
        return "{:.1f}M".format(v/1000000)
    if abs(v) > 1000:
        return "{:.1f}K".format(v/1000)
    return v


def choose_color(t0, t1):
    color_start = Colors.DARK_GRAY
    if t1 > t0 * 1.05:
        color_start = Colors.LIGHT_RED
    if t1 < t0 * 0.95:
        color_start = Colors.GREEN
    color_end = Colors.END
    return color_start, color_end


def show_compare_result(log_fileA, log_fileB, reportA, reportB):
    my_verbose_converter = set_verbose()
    pc_by_node0, pc_by_type0, stat0, verbose_by_name0, statis_by_type0 = analyse(log_fileA, reportA)
    if reportB:
        pc_by_node1, pc_by_type1, stat1, verbose_by_name1, statis_by_type1 = analyse(log_fileB, reportB)
    else:
        pc_by_node1 = []
        pc_by_type1 = []
        stat1 = []
        verbose_by_name1 = ""
        statis_by_type1 = []

    print("{}   :    {}".format(log_fileA, log_fileB))
    print("*********************************************************")
    print("*                   comparing by statis                 *")
    print("*********************************************************")
    # collect all type names
    type_names = [t for t, _ in statis_by_type0]
    for t, _ in statis_by_type1:
        if not t in type_names:
            type_names.append(t)

    def find(pclist, type_name):
        for name, v in pclist:
            if name == type_name:
                return v
        return None

    total_time0 = total_time1 = 0
    all_time0 = all_time1 = 0
    for item in statis_by_type0:
        _, v = item
        all_time0 += v[1]
    for item in statis_by_type1:
        _, v = item
        all_time1 += v[1]
    results = []
    for type_name in type_names:
        v0 = find(statis_by_type0, type_name)
        if v0:
            cnt, time0 = v0
            total_time0 += time0
            info0 = "{:.1f} x {}".format(time0/cnt, cnt)
        else:
            time0 = 0
            info0 = "---"

        v1 = find(statis_by_type1, type_name)
        if v1:
            cnt, time1 = v1
            total_time1 += time1
            info1 = "{:.1f} x {}".format(time1/cnt, cnt)
        else:
            time1 = 0
            info1 = "---"
        
        color_start, color_end = choose_color(time0, time1)
        if all_time1:
            print("{} {:>8} {:>24}({:>5.1f}%)   ({:>5.1f}%){:<24}   {} {}".format(color_start, smart_val(time1-time0), info0, time0 * 100.0 / all_time0, time1 * 100 / all_time1, info1, type_name, color_end))
            results.append((type_name, (time1 - time0) / 1000))
        else:
            print("{} {:>8} {:>24}({:>5.1f}%)   ({:>5.1f}%){:<24}   {} {}".format(color_start, smart_val(time1-time0), info0, time0 * 100.0 / all_time0, 0, info1, type_name, color_end))
            results.append((type_name, (time0)/1000))

    print("*********************************************************")
    print("*                   comparing by type                   *")
    print("*********************************************************")
    # collect all type names
    type_names = [t for t, _ in pc_by_type0]
    for t, _ in pc_by_type1:
        if not t in type_names:
            type_names.append(t)

    def find(pclist, type_name):
        for name, v in pclist:
            if name == type_name:
                return v
        return None

    total_time0 = total_time1 = 0
    results = []
    for type_name in type_names:
        v0 = find(pc_by_type0, type_name)
        if v0:
            cnt, time0 = v0
            total_time0 += time0
            info0 = "{:.1f} x {}".format(time0/cnt, cnt)
        else:
            time0 = 0
            info0 = "---"

        v1 = find(pc_by_type1, type_name)
        if v1:
            cnt, time1 = v1
            total_time1 += time1
            info1 = "{:.1f} x {}".format(time1/cnt, cnt)
        else:
            time1 = 0
            info1 = "---"
        
        color_start, color_end = choose_color(time0, time1)
        print("{} {:>8} {:>32}   {:<32}   {} {}".format(color_start, smart_val(time1-time0),  info0, info1, type_name, color_end))
        if reportB:  results.append((type_name, (time1-time0)/1000))
        else: results.append((type_name, (time0)/1000))

    color_start, color_end = choose_color(total_time0, total_time1)
    print("")
    print("{:>8}  {:>32}   {:<32}   {}".format(smart_val(total_time1 - total_time0), total_time0, total_time1, "Totals"))

    node_cnt_to_show = 10000
    node_type = ""
    show_verbose = True

    if (node_cnt_to_show > 0):
        print("*********************************************************")
        print("*                   comparing by node                   *")
        print("*********************************************************")
        # collect all type names
        all_names = [t for t, _ in pc_by_node0]
        for t, _ in pc_by_node1:
            if not t in all_names:
                all_names.append(t)

        # 
        total_time0 = total_time1 = 0
        for name in all_names:
            if (node_cnt_to_show <= 0):
                break
            v0 = find(pc_by_node0, name)
            if v0:
                time0, layer0, exectype = v0
                
                info0 = "{}_{}_{} {:6.1f}".format(layer0, exectype, find_layout(exec_graphA, name), time0)
            else:
                time0 = 0
                info0 = "---"

            v1 = find(pc_by_node1, name)
            if v1:
                time1, layer1, exectype = v1
                info1 = "{:<6.1f} {}_{}_{}".format(time1, layer1, exectype, find_layout(exec_graphB, name))
            else:
                time1 = 0
                info1 = "---"
            
            if node_type in info0 or node_type in info1:
                node_cnt_to_show -= 1
                total_time0 += time0
                total_time1 += time1
                color_start, color_end = choose_color(time0, time1)
                print("{} {:>6} {:>50}  {:<50}  {} {}".format(color_start, smart_val(time1-time0), info0, info1, name, color_end))

                if show_verbose == True:
                    verbose = ''
                    if name in verbose_by_name0:
                        verbose = 'onednn_verbose,exec,' + verbose_by_name0[name]
                        print(verbose)
                        all_verbose = \
f'''
onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
{verbose},1.7478
'''
                        if my_verbose_converter:
                            status, output = my_verbose_converter(verbose_level=0, parser='oneDNN', input=all_verbose.splitlines(), action='generate', generator='benchdnn', split_output=False, agg_keys=None)
                            if output != None:
                                for key, value in output.items():
                                    print(f"./benchdnn --fix-times-per-prb=1000 --mode=p {value}", end='')
                    if name in verbose_by_name1:
                        verbose = 'onednn_verbose,exec,' + verbose_by_name1[name]
                        print(verbose)
                        all_verbose = \
f'''
onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
{verbose},1.7478
'''
                        if my_verbose_converter:
                            status, output = my_verbose_converter(verbose_level=0, parser='oneDNN', input=all_verbose.splitlines(), action='generate', generator='benchdnn', split_output=False, agg_keys=None)
                            if output != None:
                                for key, value in output.items():
                                    print(f"./benchdnn --fix-times-per-prb=1000 --mode=p {value}", end='')
                    
        color_start, color_end = choose_color(total_time0, total_time1)
        print("")
        print("{}{:>6} {:>50}   {:<50}   {}{}".format(color_start, smart_val(total_time1 - total_time0), total_time0, total_time1, "Totals", color_end))

    print("")
    for i in range(len(stat0)):
        s0 = stat0[i].rstrip("\n").rstrip("\r")
        if stat1: s1 = stat1[i].rstrip("\n").rstrip("\r")
        else: s1 = "None"
        print("{:>50}   {:<50} ".format(s0, s1))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--node_cnt", help="number of nodes to show", default=10, type=int)
    parser.add_argument("-t","--node_type",help="node type filter",default="", type=str)
    parser.add_argument("log_fileA", nargs="?")
    parser.add_argument("log_fileB", nargs="?")
    parser.add_argument("exec_graphA", nargs="?")
    parser.add_argument("exec_graphB", nargs="?")
    parser.add_argument("-s", "--show_verbose", default=True, help="show onednn verbose", type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("-rA", "--reportA", help="report folderA", default="./a", type=str)
    parser.add_argument("-rB", "--reportB", help="report folderB", default="./b", type=str)
    args = parser.parse_args()
    with open(args.exec_graphA or 'exec_graph_A.xml') as f:
        exec_graphA = f.readlines()

    with open(args.exec_graphB or 'exec_graph_B.xml') as f:
        exec_graphB = f.readlines()

    show_compare_result(args.log_fileA or 'pcA.txt', args.log_fileB or 'pcB.txt', args.reportA, args.reportB)

#show_result(args.log_files[0], pc_by_node0, pc_by_type0, stat0)
