import argparse
import os
import csv
import json
import torch
from laion_clap import CLAP_Module

def main():
    parser = argparse.ArgumentParser(description="Evaluate MusicGen outputs using CLAP score.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run output directory (e.g., outputs/20261109_211419)")
    args = parser.parse_args()

    # 1. CLAP 모델 로드
    print("Loading CLAP model...")
    clap_model = CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt()
    clap_model.eval()

    # 2. 결과 디렉터리의 results.csv 로드
    results_csv = os.path.join(args.run_dir, "results.csv")
    if not os.path.exists(results_csv):
        print(f"Error: {results_csv} not found. Please run compare experiments first.")
        return

    scores = []
    print(f"Evaluating audio files in {args.run_dir}...")

    with open(results_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategy = row["strategy"]
            case_id = row.get("case_id") or row.get("id") or row.get("case") or list(row.values())[0]
            final_prompt = row.get("final_prompt") or row.get("prompt") or row.get("text") or row.get("description") or list(row.values())[1]
            audio_filename = row.get("wav") or row.get("audio_file") or row.get("filename") or row.get("audio_path")

            if not audio_filename:
                print(f"Skipping row due to missing audio filename. Row data: {row}")
                continue
            
            audio_path = os.path.join(args.run_dir, str(audio_filename))
            
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file {audio_path} does not exist. Skipping.")
                continue

            # 3. 텍스트 및 오디오 임베딩 추출 후 코사인 유사도 계산
            with torch.no_grad():
                text_embed = clap_model.get_text_embedding([final_prompt], use_tensor=True)
                audio_embed = clap_model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
                
                similarity = torch.cosine_similarity(text_embed, audio_embed).item()

            scores.append({
                "case_id": case_id,
                "strategy": strategy,
                "final_prompt": final_prompt,
                "audio_file": audio_filename,
                "clap_score": similarity
            })
            print(f"[{case_id}][{strategy}] CLAP Score: {similarity:.4f}")

    # 4. 요약 결과를 JSON 파일로 저장
    output_summary_path = os.path.join(args.run_dir, "clap_evaluation.json")
    with open(output_summary_path, mode="w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete! Summary saved to {output_summary_path}")

if __name__ == "__main__":
    main()