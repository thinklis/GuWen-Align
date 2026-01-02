import os
import json
import re
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # ================= 配置区域 =================
    input_root_dir = "./erya_results/json_format"
    output_root_dir = "./output/erya_DeepSeekJudge"
    model_path = "/data3/lishuo/downloads_checkpoints/deepseek-v2.5" 
    
    # vLLM 会自动吞噬大部分显存，如果显存不够，调整 gpu_memory_utilization
    # tensor_parallel_size=1 表示用 1 张卡，如果是多卡请改为卡数
    tensor_parallel_size = 8 

    # 给足显存，防止碎片化
    gpu_memory_utilization = 0.95
    # ===========================================

    print(f"正在加载 vLLM 模型: {model_path} ...")
    # vLLM 不需要像 transformers 那样手动把 model 搬到 cuda，它自己管理
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=tensor_parallel_size, 
        trust_remote_code=True,
        gpu_memory_utilization=0.95, # 显存占用比例
        dtype="bfloat16" # 或者 "float16"
    )
    
    # 只需要加载 Tokenizer 用来处理 chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 采样参数: 贪婪解码
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

    print(f"开始处理: {input_root_dir} -> {output_root_dir}")
    
    for root, dirs, files in os.walk(input_root_dir):
        for file in files:
            if not file.endswith(".json") or file.endswith("_qwen.json"):
                continue

            input_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_file_path, input_root_dir)
            output_file_path = os.path.join(output_root_dir, relative_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            print(f"正在评判: {relative_path}")
            process_single_file_vllm(llm, tokenizer, sampling_params, input_file_path, output_file_path)

    print("\n✅ 所有评判任务完成！")

def construct_prompt(src, ref, mt):
    # ... (保持原有的 Prompt 构造逻辑不变) ...
    return f"""你是一个精通古汉语和现代汉语的翻译专家。请对以下【机器翻译】的质量进行打分。

【原文（古文）】：{src}
【参考译文（现代文）】：{ref}
【机器翻译】：{mt}

请根据以下标准进行评分（0-100分）：
1. 准确性：译文是否准确表达了古文的原意，没有误译或漏译。
2. 流畅性：译文是否符合现代汉语的表达习惯，通顺易懂。
3. 信达雅：译文是否在准确的基础上，保留了原文的语气和神韵。

请直接输出一个 0 到 100 之间的分数，不需要任何解释。只输出数字即可。
分数："""

def extract_score(response):
    numbers = re.findall(r"\d+\.?\d*", response)
    if not numbers: return 0.0
    try:
        return max(0.0, min(100.0, float(numbers[-1])))
    except:
        return 0.0

def process_single_file_vllm(llm, tokenizer, sampling_params, input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [Skip] 读取失败: {e}")
        return

    if isinstance(data, dict): data = [data]
    if not isinstance(data, list): return

    # 1. 构造所有 Prompts
    prompts = []
    for item in data:
        src = item.get("input", "")
        mt = item.get("prediction", "")
        ref = item.get("target", "")
        prompt_content = construct_prompt(src, ref, mt)
        messages = [
            {"role": "system", "content": "你是一个公正的古文翻译评判助手。"},
            {"role": "user", "content": prompt_content}
        ]
        # vLLM 支持直接传入 prompt 字符串
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # 2. vLLM 极速生成 (它会自动做 Continuous Batching)
    # 只要一次性把所有 prompts 扔给它，它会尽可能塞满显卡
    outputs = llm.generate(prompts, sampling_params)

    # 3. 结果回填
    scores = []
    # vLLM 返回的顺序和 prompts 顺序是一致的
    for idx, output in enumerate(outputs):
        response = output.outputs[0].text
        score_val = extract_score(response)
        scores.append(score_val)
        
        data[idx]["qwen_score"] = score_val
        data[idx]["qwen_rationale"] = response

    # 4. 保存
    avg_score = sum(scores) / len(scores) if scores else 0
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    # 保存平均分
    file_stem = Path(output_path).stem
    parent_dir = os.path.dirname(output_path)
    avg_output_path = os.path.join(parent_dir, f"{file_stem}_qwen.json")
    
    avg_data = {
        "filename": os.path.basename(input_path),
        "sample_count": len(data),
        "average_qwen_score": avg_score,
        "model_used": "Qwen2.5-Instruct (vLLM)"
    }
    with open(avg_output_path, 'w', encoding='utf-8') as f:
        json.dump(avg_data, f, ensure_ascii=False, indent=4)
        
    print(f"  -> 平均分: {avg_score:.2f}")

if __name__ == "__main__":
    main()