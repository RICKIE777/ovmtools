import compare_vis as compare_vis


class Model():
    def __init__(self, config, logA, fpsA, logB="", fpsB=""):
        self.config = config
        self.logA = logA
        self.fpsA = fpsA
        self.logB = logB
        self.fpsB = fpsB

    def median(self, fps_res):
        v_lst = []
        for i in fps_res:
            v_lst.append(float(i.split()[0]))
        # v = v0 + v1 + v2 - max([v0, v1, v2]) - min([v0, v1, v2])
        v = max(v_lst)
        results_data = (v, fps_res[0].split()[1])
        return results_data, v_lst.index(max(v_lst))

    def filter_res(self):
        fpsA_median, _ = self.median(self.fpsA)
        if eval(self.config['Mode']['single']):
            results_lst = ','.join([f'{fpsA_median[1]},{fpsA_median[0]:.2f}'])
            result_sort_sets = (fpsA_median[1], fpsA_median[0])
            return results_lst, result_sort_sets
        fpsB_median, _ = self.median(self.fpsB)
        ratio = (fpsA_median[0] - fpsB_median[0]) / fpsB_median[0]
        geomean = fpsA_median[0] / fpsB_median[0]
        results_lst = ','.join([f'{fpsA_median[1]},{fpsA_median[0]:.2f},{fpsB_median[0]:.2f}, {ratio:.2f}, {geomean:.2f}'])
        result_sort_sets = []
        # model,prefixA,prefixB,ratio (A-B)/B,geomean

        if fpsA_median[0] < 0 or fpsB_median[0] <= 0:
            print(f'{fpsA_median[1]} has no fps, skipped.')
        if (fpsA_median[0] - fpsB_median[0]) / fpsB_median[0] < float(self.config['Filter']['threshold']):
            # model_name, prefixA_fps, prefixB_fps
            result_sort_sets = (fpsA_median[1], fpsA_median[0], fpsB_median[0])
        if not eval(self.config['Filter']['need_filter']):
            result_sort_sets = (fpsA_median[1], fpsA_median[0], fpsB_median[0])
        return results_lst, result_sort_sets

    def run_filter_compare_tool(self, save_path, reportA, reportB=""):
        result = compare_vis.show_compare_result(f'{save_path}/testA.log', f'{save_path}/testB.log', reportA, reportB)
        return result
