import os
from pathlib import Path
import subprocess
from Filter import Model
import configparser
from compare_json import CompareModel
from compare_json import Analyze
import shutil


class Process:
    def __init__(self, config):
        self.config = config
        self.fp32_model = self.config['FP32']['model_path']
        self.int8_model = self.config['INT8']['model_path']
        self.filter = eval(self.config['Filter']['need_filter']) if self.config['Filter']['need_filter'] else False
        self.benchdnn = eval(self.config['Mode']['need_run_benchdnn']) if self.config['Mode']['need_run_benchdnn'] else False
        self.compare_bin = eval(self.config['Mode']['diff_bin']) if self.config['Mode']['diff_bin'] else False
        self.compare_env = eval(self.config['Mode']['diff_env']) if self.config['Mode']['diff_env'] else False
        self.need_layer_info = True
        self.logA_detail_file = ""
        self.logB_detail_file = ""
        self.outA_file = ""
        self.outB_file = ""
        self.compare_max_file = ""
        self.filter_layer_file = ""
        self.cpus = self.config['Basic']['cpus']
        self.node = self.config['Basic']['node']
        self.script_path = os.getcwd()
        self.compare_max_lst = []
        self.filter_model_list = []
        self.verbose = ""
        self.single = eval(self.config['Mode']['single']) if self.config['Mode']['single'] else False

    def get_models(self):
        if self.fp32_model:
            if ".txt" in os.path.splitext(self.fp32_model):
                self.fp32_model = self.read_model_lst(self.fp32_model)
            else:
                self.fp32_model = Path(self.fp32_model).rglob('*.xml') if os.path.isdir(self.fp32_model) else [self.fp32_model]
        if self.int8_model:
            if ".txt" in os.path.splitext(self.int8_model):
                self.int8_model = self.read_model_lst(self.int8_model)
            else:
                self.int8_model = Path(self.int8_model).rglob('*.xml') if os.path.isdir(self.int8_model) else [self.int8_model]

    @staticmethod
    def read_model_lst(model_lst):
        with open(model_lst, "r") as fp:
            return [line.strip("\n") for line in fp]

    @staticmethod
    def del_old_data_file(save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    def openfile(self, save_path, binA_prefix, binB_prefix):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(f'{save_path}/json_data'):
            os.makedirs(f'{save_path}/json_data')
        self.logA_detail_file = open(f'{save_path}/{binA_prefix}_detail.log', 'w')
        self.outA_file = open(f'{save_path}/{binA_prefix}.log', 'w')
        self.logB_detail_file = open(f'{save_path}/{binB_prefix}_detail.log', 'w')
        self.outB_file = open(f'{save_path}/{binB_prefix}.log', 'w')
        self.filter_layer_file = open(f'{save_path}/{binA_prefix}.layer.csv', 'w')
        if not self.single:
            self.compare_max_file = open(f'{save_path}/{binA_prefix}.vs.{binB_prefix}.max.csv','w')
            compare_filter_A_B_file_title = 'model,' + f'{binA_prefix}' + ',' + f'{binB_prefix}' + ',' + "ratio (A-B)/B," + 'geomean\n'
            self.compare_max_file.write(compare_filter_A_B_file_title)
            layer_title = f'name,{binA_prefix}_fps,{binB_prefix}_fps,ratio,delta(ms),layer1,time1(ms),,,,,,,,,,\n'
            self.filter_layer_file.write(layer_title)
        else:
            self.compare_max_file = open(f'{save_path}/{binA_prefix}.max.csv', 'w')
            compare_filter_A_B_file_title = 'model,' + f'{binA_prefix}\n'
            self.compare_max_file.write(compare_filter_A_B_file_title)
            layer_title = f'name,{binA_prefix}_fps,layer1,time1(ms),,,,,,,,,,\n'
            self.filter_layer_file.write(layer_title)

    def process_file(self, precision):
        if precision == "FP32":
            save_path = f'{self.script_path}/fp32_output'
        else:
            save_path = f'{self.script_path}/int8_output'
        binA_prefix = self.config[precision]['binA_prefix']
        binB_prefix = self.config[precision]['binB_prefix']
        self.del_old_data_file(save_path)
        self.openfile(save_path, binA_prefix, binB_prefix)
        return save_path

    def closefile(self):
        self.logA_detail_file.close()
        self.outA_file.close()
        self.logB_detail_file.close()
        self.outB_file.close()
        self.compare_max_file.close()

    def process_benchmark(self):
        if self.fp32_model:
            self.compare_max_lst = []
            self.filter_model_list = []
            save_path = self.process_file(precision="FP32")
            self.choose_bins_env(self.fp32_model, "FP32", save_path)
        if self.int8_model:
            self.compare_max_lst = []
            self.filter_model_list = []
            save_path = self.process_file(precision="INT8")
            self.choose_bins_env(self.int8_model, "INT8", save_path)

    def choose_bins_env(self, models, precision, save_path):
        binA_prefix = self.config[precision]['binA_prefix']
        binB_prefix = self.config[precision]['binB_prefix']
        common_args = self.config[precision]['common_args']
        binA = self.config['BIN']['binA']
        binB = self.config['BIN']['binB']
        envA = self.config['Mode']['envA']
        envB = self.config['Mode']['envB']
        for idx, path in enumerate(models):
            model_path = os.path.normpath(str(path))
            print(f'{idx:3d} {model_path}... ', end='')
            benchmark_log_A_lst = []
            fps_resultA = []
            benchmark_log_B_lst = []
            fps_resultB = []
            if self.single:
                benchmark_log_A_lst, fps_resultA = self.run_benchmark_app(binA, model=model_path,
                                                                        common_args=common_args,
                                                                        log_file=self.logA_detail_file,
                                                                        out_file=self.outA_file,
                                                                        report_folder=f'{save_path}/a',
                                                                        exec_graph=f'{save_path}/exec_graph_A.xml')
                binB = binA
            elif self.compare_bin and self.compare_env:
                benchmark_log_A_lst, fps_resultA = self.run_benchmark_app(binA, envA, model=model_path,
                                                                        common_args=common_args,
                                                                        log_file=self.logA_detail_file,
                                                                        out_file=self.outA_file,
                                                                        report_folder=f'{save_path}/a',
                                                                        exec_graph=f'{save_path}/exec_graph_A.xml')
                benchmark_log_B_lst, fps_resultB = self.run_benchmark_app(binB, envB, model=model_path,
                                                                        common_args=common_args,
                                                                        log_file=self.logB_detail_file,
                                                                        out_file=self.outB_file,
                                                                        report_folder=f'{save_path}/b',
                                                                        exec_graph=f'{save_path}/exec_graph_B.xml')
            elif self.compare_bin and not self.compare_env:
                benchmark_log_A_lst, fps_resultA = self.run_benchmark_app(binA, model=model_path,
                                                                        common_args=common_args,
                                                                        log_file=self.logA_detail_file,
                                                                        out_file=self.outA_file,
                                                                        report_folder=f'{save_path}/a',
                                                                        exec_graph=f'{save_path}/exec_graph_A.xml')
                benchmark_log_B_lst, fps_resultB = self.run_benchmark_app(binB, model=model_path,
                                                                        common_args=common_args,
                                                                        log_file=self.logB_detail_file,
                                                                        out_file=self.outB_file,
                                                                        report_folder=f'{save_path}/b',
                                                                        exec_graph=f'{save_path}/exec_graph_B.xml')
            elif not self.compare_bin and self.compare_env:
                benchmark_log_A_lst, fps_resultA = self.run_benchmark_app(binA, envA, model=model_path,
                                                                        common_args=common_args,
                                                                        log_file=self.logA_detail_file,
                                                                        out_file=self.outA_file,
                                                                        report_folder=f'{save_path}/a',
                                                                        exec_graph=f'{save_path}/exec_graph_A.xml')
                benchmark_log_B_lst, fps_resultB = self.run_benchmark_app(binA, envB, model=model_path,
                                                                        common_args=common_args,
                                                                        log_file=self.logB_detail_file,
                                                                        out_file=self.outB_file,
                                                                        report_folder=f'{save_path}/b',
                                                                        exec_graph=f'{save_path}/exec_graph_B.xml')
                binB = binA
            m_obj = Model(self.config, benchmark_log_A_lst, fps_resultA, benchmark_log_B_lst, fps_resultB)
            if benchmark_log_A_lst:
                _, index = m_obj.median(fps_resultA)
                benchmark_log_A = benchmark_log_A_lst[index]
                self.write_benchamrk_temp_log(f'{save_path}/testA.log', benchmark_log_A)
            if benchmark_log_B_lst:
                _, index = m_obj.median(fps_resultB)
                benchmark_log_B = benchmark_log_B_lst[index]
                self.write_benchamrk_temp_log(f'{save_path}/testB.log', benchmark_log_B)
            if self.single:
                self.get_single_data(save_path, model_path, benchmark_log_A, fps_resultA, f'{save_path}/a', binA_prefix)
            elif benchmark_log_B or benchmark_log_A:
                self.get_filter_data(save_path, model_path, benchmark_log_A, fps_resultA, f'{save_path}/a', binA_prefix,
                                     benchmark_log_B, fps_resultB, f'{save_path}/b',binB_prefix)
        self.write_into_max_csv(self.compare_max_file, self.compare_max_lst)
        if not self.single and self.need_layer_info:
            self.get_analyze_data(save_path, binA_prefix, binA, binB_prefix, binB)
        if self.single and self.need_layer_info:
            self.get_analyze_data(save_path, binA_prefix, binA)
        self.closefile()

    def run_benchmark_app(self, bin, env="", model="", common_args="", log_file="", out_file="",
                        report_folder="", exec_graph=""):
        # os.chdir(bin)
        print("bin: ", bin)
        env_lst = []
        os.environ["ONEDNN_VERBOSE"] = "1"
        if self.benchdnn:
            os.environ["OV_CPU_DEBUG_LOG"] = 'CreatePrimitives;conv.cpp;deconv.cpp'
            os.environ["VERBOSE_CONVERT"] = bin + "/../../../src/plugins/intel_cpu/thirdparty/onednn/scripts/verbose_converter"
        if env: env_lst = self.process_env(env)
        result = []
        if not os.path.exists(report_folder):
            os.makedirs(report_folder)
        out_lst = []
        for i in range(int(self.config["Basic"]["run_times"])):
            log_file.write(f'========================= {bin} testing {model}...\n')
            if env_lst:
                for para, val in env_lst:
                    os.environ[para] = val
            if eval(self.config["Mode"]["numactl"]):
                bin_cmd = 'numactl -C ' + self.cpus + ' -m ' + self.node + f' {bin}/benchmark_app' + \
                          ' -m ' + model + " " + common_args + ' -exec_graph_path ' + exec_graph + \
                          ' -report_folder=' + report_folder
            else:
                bin_cmd = f' {bin}/benchmark_app' + \
                          ' -m ' + model + " " + common_args + ' -exec_graph_path ' + exec_graph + \
                          ' -report_folder=' + report_folder
            print("cmd: ", bin_cmd)
            outputA = subprocess.run(bin_cmd, shell=True, capture_output=True, env=os.environ.copy())
            out = outputA.stdout.decode()
            out_lst.append(out)
            # print("benchmark detail log: ", out)
            log_file.write(out)
            if outputA.returncode == 0:
                fps = out.split('Throughput')[-1]
                fps = fps.split()[1]
                line = f'{fps} {model}'
                result.append(line)
                print(f'{fps}', end=' ')
            else:
                line = f'-1 {model}'
                result.append(line)
            out_file.write(line + '\n')
            out_file.flush()
        print(' ')
        print("finish")
        return out_lst, result

    def process_env(self, env):
        env_tmp = env.split(",")
        env_lst = []
        for e in env_tmp:
            env_lst.append(e.split("="))
        return env_lst

    def get_analyze_data(self, save_path, binA_prefix, binA, binB_prefix="", binB=""):
        create_compare_res = Analyze(self.cpus, self.benchdnn, f'{save_path}/json_data',
                                     f'{save_path}/detail_layer_{binA_prefix}_{binB_prefix}.csv',
                                     binA_prefix, binB_prefix, binA, binB)
        create_compare_res.get_data()

    def create_each_layer_json_data(self, save_path, model_path, binA_prefix, binB_prefix=""):
        create_layer_json = CompareModel(f'{save_path}/exec_graph_A.xml', f'{save_path}/exec_graph_B.xml',
                                         model_path, f'{save_path}/testA.log', f'{save_path}/testB.log',
                                         binA_prefix, binB_prefix, f'{save_path}/a', f'{save_path}/b', save_path)
        create_layer_json.run_create_compare_csv_tool()

    def get_filter_data(self, save_path, model_path, benchmark_log_A, fps_resultA, reportA, binA_prefix,
                        benchmark_log_B, fps_resultB, reportB, binB_prefix):
        self.get_max_csv_data(save_path, model_path, benchmark_log_A, fps_resultA, reportA, binA_prefix,
                              benchmark_log_B, fps_resultB, reportB, binB_prefix)

    def get_single_data(self, save_path, model_path, benchmark_log_A, fps_resultA, reportA, binA_prefix):
        self.get_max_csv_data(save_path, model_path, benchmark_log_A, fps_resultA, reportA, binA_prefix)

    def get_max_csv_data(self, save_path, model_path, benchmark_log_A, fps_resultA, reportA, binA_prefix,
                         benchmark_log_B="", fps_resultB="", reportB="", binB_prefix=""):
        model = Model(self.config, benchmark_log_A, fps_resultA, benchmark_log_B, fps_resultB)
        results_lst, result_sort_sets = model.filter_res()
        self.compare_max_lst.append(results_lst)
        if result_sort_sets and fps_resultB:
            self.filter_model_list.append(result_sort_sets)
            layer_result = model.run_filter_compare_tool(save_path, reportA, reportB)
            self.write_layer_csv(result_sort_sets, layer_result)
            if self.need_layer_info:
                self.create_each_layer_json_data(save_path, model_path, binA_prefix, binB_prefix)
        if self.single:
            layer_result = model.run_filter_compare_tool(save_path, reportA)
            self.write_layer_csv(result_sort_sets, layer_result)
            if self.need_layer_info:
                self.create_each_layer_json_data(save_path, model_path, binA_prefix)

    def write_layer_csv(self, result_sort, layer_result):
        name = result_sort[0]
        binA_fps = result_sort[1]
        if self.single:
            self.filter_layer_file.write(f'{name},{binA_fps},')
            result = sorted(layer_result, key=lambda x: x[1], reverse=True)
            for (n, t) in result:
                self.filter_layer_file.write(f'{n},{t},')
        else:
            binB_fps = result_sort[2]
            ratio = (binA_fps-binB_fps)*100/binB_fps
            self.filter_layer_file.write(f'{name},{binA_fps},{binB_fps},{ratio:.2f}%,{(-1/binA_fps+1/binB_fps)*1000},')
            result = sorted(layer_result, key=lambda x: x[1])
            for (n, t) in result:
                if t < 0:
                    self.filter_layer_file.write(f'{n},{t},')
        self.filter_layer_file.write('\n')

    def write_benchamrk_temp_log(self, file, benchmark_log):
        with open(file, 'w') as f:
            f.write(benchmark_log)

    def write_into_max_csv(self, file, data):
        file.write("\n".join(data))


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")
    A = Process(config)
    A.get_models()
    A.process_benchmark()