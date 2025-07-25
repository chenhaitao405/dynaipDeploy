import pnnx
import torch
import os
from model.model_withoutInit import Poser_withoutInit
import utils.config as cfg
import numpy as np
import datetime


def compare_tensors(tensor1, tensor2, rtol=1e-5, atol=1e-8):
    """比较两个张量的差异并返回详细信息"""
    if isinstance(tensor1, list) and isinstance(tensor2, list):
        # 处理列表形式的输出
        if len(tensor1) != len(tensor2):
            return {
                'is_close': False,
                'error': f"列表长度不匹配: {len(tensor1)} vs {len(tensor2)}"
            }

        comparisons = []
        all_close = True

        for i, (t1, t2) in enumerate(zip(tensor1, tensor2)):
            comp = compare_tensors(t1, t2, rtol, atol)
            if not comp['is_close']:
                all_close = False
            comp['index'] = i
            comparisons.append(comp)

        return {
            'is_close': all_close,
            'comparisons': comparisons
        }

    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        abs_diff = torch.abs(tensor1 - tensor2)
        max_abs_diff = torch.max(abs_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        max_rel_diff = torch.max(torch.abs((tensor1 - tensor2) / (tensor1 + 1e-10))).item()

        # 找到差异最大的元素位置
        max_diff_idx = torch.argmax(abs_diff)
        unraveled_idx = np.unravel_index(max_diff_idx.cpu().numpy(), tensor1.shape)

        return {
            'is_close': False,
            'max_abs_diff': max_abs_diff,
            'mean_abs_diff': mean_abs_diff,
            'max_rel_diff': max_rel_diff,
            'max_diff_location': unraveled_idx,
            'value1_at_max_diff': tensor1[unraveled_idx].item(),
            'value2_at_max_diff': tensor2[unraveled_idx].item()
        }
    return {'is_close': True}


def convert_model(model, input_imu, glb_states,poser_states, output_dir,model_name):
    """使用pnnx转换模型"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 保存模型
        mod = torch.jit.trace(model, (input_imu, glb_states,poser_states))
        model_path = os.path.join(output_dir, f"model_{model_name}.pt")
        mod.save(model_path)

        # 使用pnnx转换模型
        pnnx.convert(model_path, (input_imu,glb_states,poser_states))
        print(f"模型转换成功，保存至: {model_path}")
        return True
    except Exception as e:
        print(f"模型转换失败: {str(e)}")
        return False

def move_tensors_to_cpu(nested_structure):
    """
    递归遍历嵌套结构（列表、元组、张量等），将所有张量转移到 CPU
    """
    if isinstance(nested_structure, torch.Tensor):
        # 如果是张量，直接转 CPU
        return nested_structure.cpu()
    elif isinstance(nested_structure, (list, tuple)):
        # 如果是列表/元组，递归处理每个元素
        new_elements = [move_tensors_to_cpu(elem) for elem in nested_structure]
        # 保持原数据类型（列表/元组）返回
        return type(nested_structure)(new_elements)
    else:
        # 非张量、非列表/元组的类型，直接返回（如 int、float 等）
        return nested_structure
def get_model_inference(net, bool_convert = False,model_name='onlyGLB'):
    # 加载测试用例
    test_case_dir = os.path.join(cfg.work_dir_compare, 'test_cases')
    test_case_files = [f for f in os.listdir(test_case_dir) if f.endswith('.pt')]

    if not test_case_files:
        print("未找到测试用例文件")
        return False

    # 选择最新的测试用例
    test_case_file = os.path.join(test_case_dir, test_case_files[0])
    test_case = torch.load(test_case_file)

    # 提取数据
    input_imu = test_case['input_imu']
    v_init = test_case['v_init']
    p_init = test_case['p_init']
    v_out_baseline = [tensor.cpu() for tensor in test_case['v_out_baseline']]
    p_out_baseline = [tensor.cpu() for tensor in test_case['p_out_baseline']]
    states = torch.load("states7.pth")
    states = move_tensors_to_cpu(states)
    glb_states =  torch.stack(states[0], dim=0)

    all_tensors = []
    for tup in states[1:]:
        all_tensors.extend(tup)  # 每个元组的 2 个 tensor 加入列表
    # 2. 沿维度 1 拼接（dim=1），得到 shape=(2,12,200) 的张量
    poser_states = torch.stack(all_tensors, dim=0)
    # 执行模型转换（如果需要）
    if bool_convert:
        convert_model(net, input_imu, glb_states, poser_states,
                      os.path.join(cfg.work_dir_compare, 'converted_models'), model_name)

    # 运行当前模型
    with torch.no_grad():
        v_out_current,p_out_current ,states= net.forward(input_imu, glb_states, poser_states)


    # 比较输出
    v_comparison = compare_tensors([v_out_current], v_out_baseline)
    p_comparison = compare_tensors([p_out_current], p_out_baseline)

    # 输出比较结果
    print(f"\n测试用例: {os.path.basename(test_case_file)}")
    print(f"基准模型权重: {test_case['metadata']['model_weights']}")

    def print_comparison_result(name, comparison):
        if comparison['is_close']:
            print(f"\n{name}输出比较:")
            print(f"✅ 当前模型和基准模型的{name}输出一致")
        else:
            print(f"\n{name}输出比较:")
            if 'error' in comparison:
                print(f"❌ 比较失败: {comparison['error']}")
                return

            if 'comparisons' in comparison:
                for comp in comparison['comparisons']:
                    if comp['is_close']:
                        print(f"  ✅ 列表元素 {comp['index']} 一致")
                    else:
                        print(f"  ❌ 列表元素 {comp['index']} 存在差异")
                        print(f"    最大绝对差异: {comp['max_abs_diff']:.6f}")
                        print(f"    平均绝对差异: {comp['mean_abs_diff']:.6f}")
                        print(f"    最大相对差异: {comp['max_rel_diff']:.6f}")
                        print(f"    最大差异位置: {comp['max_diff_location']}")
                        print(f"    基准值: {comp['value1_at_max_diff']:.6f}")
                        print(f"    当前值: {comp['value2_at_max_diff']:.6f}")
            else:
                print(f"❌ 当前模型和基准模型的{name}输出存在差异")
                print(f"  最大绝对差异: {comparison['max_abs_diff']:.6f}")
                print(f"  平均绝对差异: {comparison['mean_abs_diff']:.6f}")
                print(f"  最大相对差异: {comparison['max_rel_diff']:.6f}")
                print(f"  最大差异位置: {comparison['max_diff_location']}")
                print(f"  基准值: {comparison['value1_at_max_diff']:.6f}")
                print(f"  当前值: {comparison['value2_at_max_diff']:.6f}")

    print_comparison_result("速度", v_comparison)
    print_comparison_result("姿态", p_comparison)


def main(model = 'PT',bool_convert=False,model_name='onlyGLB'):
    """主函数：加载模型、测试用例，运行模型并比较输出"""
    # 加载模型
    if model =='pt':
        # 假设保存的模型路径
        output_dir = os.path.join(cfg.work_dir_compare, 'converted_models')
        model_path = os.path.join(output_dir, f"model_{model_name}.pt")
        net = torch.jit.load(model_path)
        net.eval()
    else:
        net = Poser_withoutInit()
        net.load_state_dict(torch.load(cfg.weight_s_compare, map_location='cpu'))
        net.eval()

    #对比结果
    get_model_inference(net,bool_convert,model_name)

    return True


if __name__ == "__main__":
    # 默认不进行模型转换，如需转换请设置为True
    main(model ='pth',bool_convert=True, model_name = 'withoutInit')